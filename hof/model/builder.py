from copy import deepcopy
import torch
from hof.utils.registry import Registry

MODEL = Registry("model")

def build_model(cfg) -> torch.nn.Module:
    # Import only the architectures needed by HO3D inference.
    from .backbones import dinov2, ptv3
    from .heads import mano_head
    from . import hoi_denoiser, hoi_flow
    return MODEL.build(deepcopy(cfg))
