from .transformer import TransformerModel
from .k_point_selector import K_Point_Selector
import torch.nn as nn




class all_model(nn.Module):
    def __init__(self,
                input_channels,
                condition_channels,
                num_channels,
                out_channels,
                use_checkpoint,
                learn_sigma,
                diffusion_steps,
                noise_schedule,
                timestep_respacing,
                use_kl,
                predict_xstart,
                rescale_timesteps,
                rescale_learned_sigmas,
                dataset,
                set_name,
                rotation,
                exp_name,
                use_image_features,
                use_geometric_features,
                output_channels=128,
                hidden_channels=3,
                n_layers=2,
                min_score = None,
                ratio = 0.1,
                freeze_backbone=True, 
                backbone_weights_path=None) -> None:
  

    
    
        self.model1 = K_Point_Selector(input_channels, output_channels, hidden_channels, n_layers, min_score, ratio, 
                                       freeze_backbone=True, backbone_weights_path=None )
        self.model2 = TransformerModel(input_channels, condition_channels, num_channels, out_channels, use_checkpoint, rotation, use_image_features, use_geometric_features)
        


    def forward(self,  x, timesteps):


        x, nodes, score, edge_index, new_batch, batch = self.model1(x)

        x, atts = self.model1( x, timesteps)

        return x
