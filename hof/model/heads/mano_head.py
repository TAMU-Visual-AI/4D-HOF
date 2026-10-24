import torch
import torch.nn as nn
from easydict import EasyDict as edict

from .components.grasp_dit import GraspSeqRefineDiT
from .components.time_embedding import Timesteps
from ..builder import MODEL

@MODEL
class HOISeqRefineHead(nn.Module):
    def __init__(self, grasp_dit: edict, npose: int = 108):
        super().__init__()

        self.dit = GraspSeqRefineDiT(
            **grasp_dit
        )

        pose_dim = grasp_dit["pose_dim"]
        hoi_dim = grasp_dit["hoi_dim"]
        self.encpose = nn.Linear(npose, pose_dim)
        self.decpose = nn.Linear(pose_dim, npose)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)

        self.pe = Timesteps(num_channels=hoi_dim, flip_sin_to_cos=False, downscale_freq_shift=1)

    def forward(self, x, timestep, hoi_feats, visual_feats, obj_feats=None):
        # We have RoPE in flash_attention, no extra pe here.
        x = self.encpose(x)

        # Add positional encoding to 3D features
        if hoi_feats is not None:
            seq_len = hoi_feats.shape[1]
            pos = torch.arange(seq_len, dtype=torch.float32, device=hoi_feats.device)
            obj_pe = self.pe(pos)
            hoi_feats = hoi_feats + obj_pe.unsqueeze(0)

        # Pass through transformer
        token_out = self.dit(x, timestep, hoi_feats, visual_feats, obj_feats)
        out = self.decpose(token_out)

        return out
