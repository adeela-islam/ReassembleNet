import torch.nn as nn
import torch
from .GCN import GCN
from .Transformer_GNN import Transformer_GNN
from .pooling import TopKPooling
from torchvision.transforms.functional import rotate
import torch_geometric
from torch_geometric.nn import aggr


class ClampToOneSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # In the forward pass, output a tensor full of ones
        return torch.ones_like(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        # In the backward pass, pass the gradient through unchanged.
        # (This is one common STE strategy; you can customize it as needed.)
        return grad_output


class Transformer_student(nn.Module):
    """
    This model contain the GNN backbone and the final MLP for the keypoints selection task.


    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        input_channels=3,
        output_channels=128,
        hidden_channels=3,
        n_layers=2,
        min_score = None,
        ratio = 0.1,
        #virt_nodes=4,
        ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.ratio = ratio
        self.min_score = min_score
        self.hidden_channels = hidden_channels


        #if architecture == "transformer":
        self.gnn_backbone1 = Transformer_GNN(
                self.hidden_channels,
                n_layers=n_layers,
                hidden_dim=self.hidden_channels* 8,
                heads=8,
                output_size=self.hidden_channels,
            )
        # self.gnn_backbone2 = Transformer_GNN(
        #         self.hidden_channels * 8,
        #         n_layers=n_layers,
        #         hidden_dim=self.hidden_channels * 8,
        #         heads=8,
        #         output_size=self.hidden_channels,
        #     )
        # self.gnn_backbone = GCN(
        #         self.input_channels,
        #         hidden_dim=32 * 8,
        #         output_size=self.input_channels,
        #     )
        self.TopKPooling = TopKPooling(self.hidden_channels, ratio=self.ratio, min_score=self.min_score)

        # self.sum_aggr =aggr.SumAggregation()

        
        # self.final_mlp = nn.Sequential(
        #     nn.Linear(self.hidden_channels, self.output_channels)#,
        #     #nn.ReLU(),
        #     #nn.Linear(32, input_channels),
        #     #nn.ReLU()
        # )
        self.mlp = nn.Sequential(
            nn.Linear(self.input_channels, 32),
            #nn.ReLU(),
            #nn.Linear(32, self.hidden_channels),
        )


        
    def forward(self, xy_pos, edge_index, batch):
        # MLP
        combined_feats = self.mlp(xy_pos) # da modificare
        
        
        # Transform with all the connections
        feats, _ = self.gnn_backbone1(x=combined_feats, batch=batch, edge_index=edge_index)

        #feats, _ = self.gnn_backbone2(x=feats, batch=batch, edge_index=edge_index)


        # Pooling
        feats, edge_index, _, batch, nodes, score = self.TopKPooling(feats, edge_index, None, batch)


        clamped_score = ClampToOneSTE.apply(score.view(-1, 1))
        xy_pos = xy_pos * clamped_score  # This multiplies by ones in the forward pass.
        x = xy_pos[nodes]

      
        return x, nodes, score, edge_index, batch #attentions


class K_Point_Selector(nn.Module):
    def __init__(
        self,
        max_num_pieces, 
        input_channels=3,
        output_channels=128,
        hidden_channels=3,
        n_layers=2,
        min_score = None,
        ratio = 0.1,
        freeze_backbone=True,
        backbone_weights_path=None  # Add a parameter for the weights path
        #virt_nodes=4,
        ) -> None:

        super(K_Point_Selector, self).__init__()

        # load the weight of the backbone
        self.nets = Transformer_student(input_channels, output_channels, hidden_channels, n_layers, min_score, ratio)
        if backbone_weights_path:
            self.nets.load_state_dict(torch.load(backbone_weights_path))
            print(f"Loaded weights from {backbone_weights_path}")
        #self.freeze_backbone = freeze_backbone
        # Freeze the backbone if not trainable
        if freeze_backbone:
            for param in self.nets.parameters():
                param.requires_grad = False
            print("Backbone model frozen.")
        self.max_num_pieces = max_num_pieces
        self.ratio = ratio

    def forward(self, x):
        B, K, features = x.size()
        T = K//self.max_num_pieces # number of keypoints for each singular graph
        batch_size = B * self.max_num_pieces  # Total number of singular graphs 

        # dimesion of x is [B, K, features] to [B *max_num_pieces, max_number_of_keypoints, features]
        # where max_number_of_keypoints = K//max_num_pieces and B*max_num_pieces are all the singular graphs
        x = x.reshape(batch_size*T, features)

        # Now I have to create the batch list
        batch = torch.arange(batch_size).repeat_interleave(T).to(x.device)


        # Create fully connected graph for a single graph (T nodes)
        edge_index = torch.cartesian_prod(torch.arange(T), torch.arange(T)).T.repeat(1, batch_size)  # Shape: [2, batch_size * T*T]

        # Compute offset for each batch graph
        offset = torch.arange(batch_size) * T  # Offset per graph
        offset = offset.repeat_interleave(T * T).repeat(2, 1)  # Match edge count and create [2, B*T*T]

        # Apply offset to edge index
        edge_index = edge_index + offset

        # Ensure all tensors have the same dtype
        edge_index = edge_index.to(x.device, dtype=torch.long)
        x = x.to(x.device, dtype=torch.float)
        batch = batch.to(x.device, dtype=torch.long)
        # Now I have to process the singular graphs
        x, nodes, score, edge_index, new_batch = self.nets(x, edge_index, batch)
        # Now x has shape [B*max_num_pieces*ratio, features]
        #x = x.reshape(B, self.max_num_pieces*self.ratio, features)

        return x, nodes, score, edge_index, new_batch, batch
    



