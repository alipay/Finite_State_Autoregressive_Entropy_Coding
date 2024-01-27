import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def batched_cross_entropy(input : torch.Tensor, target : torch.Tensor,
        min : float = 0, 
        max : float = 1,
        eps : float = 1e-6,
    ):
    """Batched Cross Entropy

    Args:
        input (torch.Tensor): B*(C*N)*H*W (N=classes)
        target (torch.Tensor): B*C*H*W
        min (float, optional): min value of target, used for normalizing. Defaults to 0.
        max (float, optional): max value of target, used for normalizing. Defaults to 1.
        eps (float, optional): a small number

    Returns:
        Per element cross entropy: B*C*H*W
    """    
    # convert to BHWC for cross_entropy
    # assert(target.ndim == 4)
    # B, C, H, W = target.shape
    assert input.shape[2:] == target.shape[2:]
    target_shape = target.shape
    batch_size = target_shape[0]
    channel_size = target_shape[1]
    spatial_dim = np.prod(target_shape[2:])
    num_symbols = input.shape[1] // target.shape[1]
    x_logits = input.reshape(batch_size, channel_size, num_symbols, spatial_dim).permute(0, 1, 3, 2).contiguous()
    
    # discretize x_target
    # TODO: check discretize method (may cause nan issue!)
    x_norm = (target.clamp(min, max) - min) / (max - min)
    x_target = (x_norm * (num_symbols - 1)).long() # .detach()
    x_ce = F.cross_entropy(
        x_logits.reshape(-1, num_symbols), 
        x_target.reshape(-1), 
        reduction="none"
    ).reshape(*target_shape)
    return x_ce #.view(input.shape[0], -1).mean(dim=1)
