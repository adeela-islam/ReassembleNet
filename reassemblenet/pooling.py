'''
Example code for testing the key points selection algorithm
'''

import torch
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor, sigmoid
from torch.nn import Parameter

from torch_geometric.nn.inits import uniform
from torch_geometric.utils import scatter, softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes


def topk(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
    min_score: Optional[float] = None,
    tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)
        perm = (x > scores_min).nonzero().view(-1)

    elif ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')
        batch_size, max_num_nodes = num_nodes.size(0), int(num_nodes.max())

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ), -60000.0)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
            k = torch.min(k, num_nodes)
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        if isinstance(ratio, int) and (k == ratio).all():
            # If all graphs have exactly `ratio` or more than `ratio` entries,
            # we can just pick the first entries in `perm` batch-wise:
            index = torch.arange(batch_size, device=x.device) * max_num_nodes
            index = index.view(-1, 1).repeat(1, ratio).view(-1)
            index += torch.arange(ratio, device=x.device).repeat(batch_size)
        else:
            # Otherwise, compute indices per graph:
            index = torch.cat([
                torch.arange(k[i], device=x.device) + i * max_num_nodes # type: ignore
                for i in range(batch_size)
            ], dim=0)

        perm = perm[index]

    else:
        raise ValueError("At least one of 'min_score' and 'ratio' parameters "
                         "must be specified")

    return perm


def filter_adj(
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    perm: Tensor,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index[0], edge_index[1]
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr


class TopKPooling(torch.nn.Module):
    r""":math:`\mathrm{top}_k` pooling operator from the `"Graph U-Nets"
    """
    def __init__(
        self,
        in_channels: int,
        ratio: Union[int, float] = 0.5,
        min_score: Optional[float] = None,
        multiplier: float = 1.,
        nonlinearity: Union[str, Callable] = 'tanh',
    ):
        super().__init__()

        if isinstance(nonlinearity, str):
            nonlinearity = getattr(torch, nonlinearity)

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.weight = Parameter(torch.Tensor(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        uniform(self.in_channels, self.weight)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        attn: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:
        r"""
        Args:
            x (torch.Tensor): The node feature matrix.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example. (default: :obj:`None`)
            attn (torch.Tensor, optional): Optional node-level matrix to use
                for computing attention scores instead of using the node
                feature matrix :obj:`x`. (default: :obj:`None`)
        """
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = (attn * self.weight).sum(dim=-1)

        if self.min_score is None:
            score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1)) # type: ignore
        else:
            #score = softmax(score, batch)
            score = sigmoid(score)
        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'

        return (f'{self.__class__.__name__}({self.in_channels}, {ratio}, '
                f'multiplier={self.multiplier})')