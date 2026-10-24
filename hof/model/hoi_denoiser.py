from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_max
from easydict import EasyDict as edict

from .mano_wrapper import MANO
from .builder import MODEL, build_model


@MODEL
class HOISeqRefineDenoiser(nn.Module):
    def __init__(self,
        cfg: edict,
        img_backbone_cfg: edict,
        hoi_encoder_cfg: edict,
        grasp_head_cfg: edict,
        obj_encoder_cfg: edict = None
        ):
        super().__init__()

        self.cfg = cfg
        self.obj_grid_size = nn.Parameter(torch.tensor(cfg.obj_grid_size, dtype=torch.float32), requires_grad=False)
        self.seq_len = cfg.seq_len
        self.pose_num = 16
        self.pose_dim = 6 # rot6d representation

        if img_backbone_cfg is not None:
            self.img_backbone = build_model(img_backbone_cfg)

        if hoi_encoder_cfg is not None:
            self.hoi_encoder = build_model(hoi_encoder_cfg)

        self.hoi_head = build_model(grasp_head_cfg)

        if img_backbone_cfg is not None and self.cfg.get("ray_encoding", False):
            visual_dim = grasp_head_cfg["grasp_dit"]["visual_dim"]

            self.plucker_proj = nn.Sequential(
                nn.Linear(6, 64),
                nn.GELU(),
                nn.Linear(64, visual_dim)
            )
            # Xavier uniform for first layer
            nn.init.xavier_uniform_(self.plucker_proj[0].weight, gain=0.01)
            nn.init.zeros_(self.plucker_proj[0].bias)
            # Zero init for the final layer so it doesn't affect pretrained features initially
            nn.init.zeros_(self.plucker_proj[2].weight)
            nn.init.zeros_(self.plucker_proj[2].bias)

        if obj_encoder_cfg is not None:
            self.obj_encoder = build_model(obj_encoder_cfg)

        # Instantiate MANO model
        self.mano = MANO(**cfg.mano_cfg)

        # Cache for conditioning features during generation
        self._cached_features = None

    def clear_cache(self):
        self._cached_features = None

    def obj_encoding(self, obj_metric_pc: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, 'obj_encoder'):
            return None

        batch_size, seq_len, point_num = obj_metric_pc.shape[0], obj_metric_pc.shape[1], obj_metric_pc.shape[2]

        feat = obj_metric_pc[:, 0] # One frame is enough

        batch = torch.cat([torch.full((point_num,), i, dtype=torch.long, device=obj_metric_pc.device) for i in range(batch_size)], dim=0)

        data_dict = {
            "coord": obj_metric_pc[:, 0].clone().reshape(batch_size * point_num, 3).contiguous(),
            "grid_size": self.obj_grid_size,
            "feat": feat.reshape(batch_size * point_num, feat.shape[-1]).contiguous(),
            "batch": batch
        }

        out = self.obj_encoder(data_dict)
        obj_feats = out["feat"]
        obj_batch = out["batch"]

        obj_feats, _ = scatter_max(obj_feats, obj_batch, dim=0, dim_size=batch_size)
        obj_feats = obj_feats.reshape(batch_size, 1, -1) # (B, 1, feat_dim)
        obj_feats = obj_feats.expand(-1, seq_len, -1) # (B, seq_len, feat_dim)

        return obj_feats

    def joint_encoding(self, verts_object: torch.Tensor, refine_vertices: torch.Tensor, refine_joints: torch.Tensor = None) -> torch.Tensor:
        if not hasattr(self, 'hoi_encoder'):
            return None

        batch_size, seq_len, obj_point_num = verts_object.shape[0], verts_object.shape[1], verts_object.shape[2]
        hand_point_num = refine_vertices.shape[2]
        point_num = obj_point_num + hand_point_num

        # Padding a cls label: 1 for obj, 0 for hand
        obj_cls = torch.ones((batch_size, seq_len, obj_point_num, 1), device=verts_object.device)
        hand_cls = torch.zeros((batch_size, seq_len, hand_point_num, 1), device=verts_object.device)

        if refine_joints is not None:
            # Euclidean distances to all 21 MANO joints
            # (B, seq_len, num_points, 3) and (B, seq_len, 21, 3) -> (B, seq_len, num_points, 21)
            obj_joint_dists = torch.norm(verts_object.unsqueeze(-2) - refine_joints.unsqueeze(-3), dim=-1)
            hand_joint_dists = torch.norm(refine_vertices.unsqueeze(-2) - refine_joints.unsqueeze(-3), dim=-1)

            # Concatenate features: 3 (coord) + 21 (joint_dists) + 1 (cls) = 25 dims
            obj_feat = torch.cat([verts_object, obj_joint_dists, obj_cls], dim=-1)
            hand_feat = torch.cat([refine_vertices, hand_joint_dists, hand_cls], dim=-1)

        else:
            # Concatenate features: 3 (coord) + 1 (cls) = 4 dims
            obj_feat = torch.cat([verts_object, obj_cls], dim=-1)
            hand_feat = torch.cat([refine_vertices, hand_cls], dim=-1)

        feat = torch.cat([obj_feat, hand_feat], dim=-2) # (B, seq_len, point_num, feat_dim)

        batch = torch.cat([torch.full((point_num,), i, dtype=torch.long, device=verts_object.device) for i in range(batch_size * seq_len)], dim=0)

        coord = feat[..., :3].clone()

        data_dict = {
            "coord": coord.reshape(batch_size * seq_len * point_num, 3),
            "grid_size": self.obj_grid_size,
            "feat": feat.reshape(batch_size * seq_len * point_num, feat.shape[-1]),
            "batch": batch
        }

        out = self.hoi_encoder(data_dict)
        joint_feats = out["feat"]
        joint_batch = out["batch"]

        joint_feats, _ = scatter_max(joint_feats, joint_batch, dim=0, dim_size=batch_size * seq_len)
        joint_feats = joint_feats.reshape(batch_size, seq_len, -1) # (B, seq_len, feat_dim)
        return joint_feats

    def ray_direction_encoding(
        self,
        img: torch.Tensor,          # (B, T, C, H, W)
        orig_center: torch.Tensor,  # (B, T, 2)  principal point (cx, cy) in original image space
        focal_length: torch.Tensor, # (B, T)     focal length in original image space (fx == fy)
        patch_size: int = 14,
    ) -> torch.Tensor:
        """
        Compute per-patch ray direction encodings for a static camera.
        Includes one extra token for the [CLS] token (mapped to image centre).

        Since the camera is static (origin at world zero), Plucker moments
        are always zero, so we only encode the 3D ray direction, padded to
        6D as (d, 0) for interface compatibility.

        Args:
            img:          Input tensor,              shape (B, T, C, H, W)
            orig_center:  Principal point (cx, cy)
                        in original image space,   shape (B, T, 2)
            focal_length: Focal length (fx == fy)
                        in original image space,   shape (B, T)
            patch_size:   Size of each image patch (e.g. 14)

        Returns:
            ray_encoding: shape (B*T, num_patches + 1, 6)
                        first 3 dims are unit ray direction, last 3 are zeros
        """
        if not hasattr(self, "plucker_proj"):
            return None

        B, T = img.shape[0], img.shape[1]
        H, W = img.shape[3], img.shape[4]
        assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch size"
        num_patches = (H // patch_size) * (W // patch_size)  # e.g. 196
        grid_size = int(np.sqrt(num_patches))  # 16

        # ------------------------------------------------------------------ #
        # 1. Patch-centre coordinates in resized (H, W) space                 #
        # ------------------------------------------------------------------ #
        patch_h = H / grid_size
        patch_w = W / grid_size

        y, x = torch.meshgrid(
            torch.arange(grid_size, device=img.device),
            torch.arange(grid_size, device=img.device),
            indexing='ij',
        )
        u_patch = (x + 0.5) * patch_w  # (grid_size, grid_size)
        v_patch = (y + 0.5) * patch_h

        # (B, T, N)
        u_patch = u_patch.flatten().view(1, 1, -1).expand(B, T, -1)
        v_patch = v_patch.flatten().view(1, 1, -1).expand(B, T, -1)

        # CLS token → image centre in resized space
        u_cls = torch.full((B, T, 1), W / 2.0, device=img.device)
        v_cls = torch.full((B, T, 1), H / 2.0, device=img.device)

        # Append CLS: (B, T, N+1)
        u_all = torch.cat([u_patch, u_cls], dim=2)
        v_all = torch.cat([v_patch, v_cls], dim=2)

        # ------------------------------------------------------------------ #
        # 2. Map resized coords back to original image space                  #
        # ------------------------------------------------------------------ #
        orig_cx = orig_center[..., 0:1]  # (B, T, 1)
        orig_cy = orig_center[..., 1:2]  # (B, T, 1)
        orig_w  = orig_cx * 2.0
        orig_h  = orig_cy * 2.0

        u_orig = u_all * (orig_w / W)    # (B, T, N+1)
        v_orig = v_all * (orig_h / H)

        # ------------------------------------------------------------------ #
        # 3. Unproject through original intrinsics                            #
        # ------------------------------------------------------------------ #
        f = focal_length.unsqueeze(-1)   # (B, T, 1)

        d_x = (u_orig - orig_cx) / f
        d_y = (v_orig - orig_cy) / f
        d_z = torch.ones_like(d_x)

        D = torch.stack([d_x, d_y, d_z], dim=-1)   # (B, T, N+1, 3)
        D = F.normalize(D, p=2, dim=-1)

        # ------------------------------------------------------------------ #
        # 4. Pad to 6D (moment = 0 for static camera)                        #
        # ------------------------------------------------------------------ #
        ray_encoding = torch.cat([D, torch.zeros_like(D)], dim=-1)  # (B, T, N+1, 6)

        return self.plucker_proj(ray_encoding)  # (B, T, N+1, feat_dim)

    def visual_encoding(self, img: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, 'img_backbone'):
            return None
        batch_size, seq_len = img.shape[0], img.shape[1]
        img = img.reshape(batch_size * seq_len, *img.shape[2:]) # (B * seq_len, C, H, W)

        img_out = self.img_backbone(img, is_training=True)
        img_patchtokens = img_out["x_norm_patchtokens"]
        img_cls_token = img_out["x_norm_clstoken"].reshape(batch_size * seq_len, 1, -1)
        visual_feats = torch.cat([img_patchtokens, img_cls_token], dim=1) # (B * seq_len, num_patches + 1, feat_dim)
        visual_feats = visual_feats.reshape(batch_size, seq_len, *visual_feats.shape[1:]) # (B, seq_len, num_patches + 1, feat_dim)

        return visual_feats

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, **batch: Dict) -> Dict:
        if not self.training and self._cached_features is not None:
            hoi_feats = self._cached_features["hoi_feats"]
            visual_feats = self._cached_features["visual_feats"]
            obj_feats = self._cached_features["obj_feats"]
        else:
            # Compute hoi features
            verts_object = batch["obj_pc_refine"]
            refine_vertices = batch["refine_vertices"]
            refine_joints = batch["refine_joints"]
            hoi_feats = self.joint_encoding(verts_object, refine_vertices, refine_joints)

            # Compute visual features
            img = batch["img"] # (B, seq_len, C, H, W)
            visual_feats = self.visual_encoding(img)

            orig_center = batch["orig_center"]  # (B, seq_len, 2)
            orig_focal_length = batch["orig_focal_length"]  # (B, seq_len)
            ray_feats = self.ray_direction_encoding(img, orig_center, orig_focal_length) # (B, seq_len, num_patches + 1, feat_dim)
            if visual_feats is not None and ray_feats is not None:
                visual_feats = visual_feats + ray_feats

            # Compute object features if obj_encoder is available
            obj_metric_pc = batch["obj_metric_pc"] # (B, seq_len, N_obj, 3)
            obj_feats = self.obj_encoding(obj_metric_pc)

            if not self.training:
                self._cached_features = {
                    "hoi_feats": hoi_feats,
                    "visual_feats": visual_feats,
                    "obj_feats": obj_feats
                }

        out = self.hoi_head(x, timestep, hoi_feats, visual_feats, obj_feats)

        return out
