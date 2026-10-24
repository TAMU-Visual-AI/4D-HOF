from typing import Dict

import torch
import torch.nn as nn

from hof.utils.transform import rotation_6d_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle
from .guidance.object_sdf import ObjectSDFCache
from .guidance.silhouette import silhouette_loss

from .diffusions.brownian import create_bridge
from .builder import MODEL, build_model


@MODEL
class HOISeqBridgeFlow(nn.Module):
    def __init__(self, bridge_cfg, denoiser_cfg, hand_dim=99, obj_dim=9, trans_delta=None, gaussian=False, hand_only=False, object_only=False, guidance_cfg=None, inference_steps=50):
        super().__init__()

        self.denoiser = build_model(denoiser_cfg)
        self.hand_dim = hand_dim
        self.obj_dim = obj_dim
        assert trans_delta in [None, "o2h", "h2o"]
        self.trans_delta = trans_delta
        self.gaussian = gaussian
        self.hand_only = hand_only
        self.object_only = object_only
        self.guidance_cfg = guidance_cfg
        self.inference_steps = inference_steps
        self._guidance_sdf_cache = ObjectSDFCache()

        self.bright_matching = create_bridge(**bridge_cfg)

    @property
    def device(self):
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Guidance helpers
    # ------------------------------------------------------------------

    @staticmethod
    def pose_smoothness_loss(
        translation: torch.Tensor,
        rotation: torch.Tensor,
        hand_pose=None,
        translation_weight=1.0,
        rotation_weight=0.05,
        frame_indices=None,
    ) -> torch.Tensor:
        """Penalize linear and angular velocity and acceleration without ground truth.

        Inputs have leading dimensions (batch, time). Translations are in meters
        and rotations are matrices; optional hand_pose contains local joint
        rotations. Angular velocity uses consecutive relative rotations mapped
        to axis-angle; its first difference gives angular acceleration.

        Average squared velocity and acceleration over components (and finger
        joints), then sum over temporal pairs/triplets and batches. Both terms
        use the same translation/rotation weights, with no balancing factor.
        Each frame's guidance strength is independent of chunk length. Time is
        measured in sampled frames. With frame_indices, velocity uses increasing
        pairs and acceleration uses equally spaced, increasing triplets,
        excluding repeated padding frames.
        """
        if translation_weight < 0 or rotation_weight < 0:
            raise ValueError("Smoothness weights must be non-negative")

        loss = translation.sum() * 0.0
        if translation.shape[1] < 2:
            return loss

        valid_velocity = valid_acceleration = None
        if frame_indices is not None:
            frame_indices = frame_indices.to(translation.device)
            if frame_indices.shape != translation.shape[:2]:
                raise ValueError("frame_indices must have shape (batch, time)")
            steps = frame_indices[:, 1:] - frame_indices[:, :-1]
            valid_velocity = steps > 0
            valid_acceleration = (steps[:, :-1] > 0) & (steps[:, :-1] == steps[:, 1:])

        def reduce(values, valid):
            per_frame = values.square().flatten(start_dim=2).mean(dim=-1)
            if valid is not None:
                per_frame = per_frame * valid
            return per_frame.sum()

        def angular_velocity(rotations):
            relative = rotations[:, :-1].transpose(-1, -2) @ rotations[:, 1:]
            return matrix_to_axis_angle(relative)

        if translation_weight > 0:
            velocity = translation[:, 1:] - translation[:, :-1]
            acceleration = translation[:, 2:] - 2.0 * translation[:, 1:-1] + translation[:, :-2]
            loss = loss + translation_weight * (
                reduce(velocity, valid_velocity) + reduce(acceleration, valid_acceleration)
            )
        if rotation_weight > 0:
            for rotations in (rotation, hand_pose):
                if rotations is not None:
                    velocity = angular_velocity(rotations)
                    acceleration = velocity[:, 1:] - velocity[:, :-1]
                    loss = loss + rotation_weight * (
                        reduce(velocity, valid_velocity) + reduce(acceleration, valid_acceleration)
                    )
        return loss

    def _guidance_gradients(
        self, x, interaction_loss, smoothness_loss, interaction_target="object",
        silhouette_loss=None,
    ):
        """Guide interaction translations, silhouette object pose, and full smoothness.

        interaction_target selects "hand", "object", or "both" independently
        of the model's hand_only/object_only state layout. Fixed components
        outside the sampled state cannot receive updates.
        """
        if interaction_target not in ("hand", "object", "both"):
            raise ValueError('interaction_target must be "hand", "object", or "both"')

        grad = torch.zeros_like(x)
        has_silhouette = silhouette_loss is not None and silhouette_loss.requires_grad and not self.hand_only
        if interaction_loss.requires_grad:
            interaction_grad = torch.autograd.grad(
                interaction_loss, x, retain_graph=smoothness_loss.requires_grad or has_silhouette,
                allow_unused=True,
            )[0]
            if interaction_grad is not None:
                # These flags describe the sampled state layout, not the target.
                if not self.object_only and interaction_target in ("hand", "both"):
                    grad[..., self.hand_dim - 3:self.hand_dim] = (
                        interaction_grad[..., self.hand_dim - 3:self.hand_dim]
                    )
                if not self.hand_only and interaction_target in ("object", "both"):
                    obj_start = 0 if self.object_only else self.hand_dim
                    grad[..., obj_start + 6:obj_start + self.obj_dim] = (
                        interaction_grad[..., obj_start + 6:obj_start + self.obj_dim]
                    )
        if has_silhouette:
            silhouette_grad = torch.autograd.grad(
                silhouette_loss, x, retain_graph=smoothness_loss.requires_grad,
                allow_unused=True,
            )[0]
            if silhouette_grad is not None:
                obj_start = 0 if self.object_only else self.hand_dim
                grad[..., obj_start:obj_start + self.obj_dim] += silhouette_grad[..., obj_start:obj_start + self.obj_dim]
        if smoothness_loss.requires_grad:
            smoothness_grad = torch.autograd.grad(smoothness_loss, x, allow_unused=True)[0]
            if smoothness_grad is not None:
                grad = grad + smoothness_grad
        return grad.detach()

    def clear_guidance_sdf_cache(self):
        """Release cached SDF volumes (also automatically replaced for new meshes)."""
        self._guidance_sdf_cache.clear()

    def _sdf_interaction_losses(self, vertices, hand_trans, obj_rot, obj_trans, batch, cfg):
        """One SDF query for contact and penetration, with translation-only gradients.

        Meshes are object-local, in meters, with the same scale/origin/axes as
        obj_metric_pc. Contact retains the binary vertex mask and squared-distance
        reduction. Penetration measures hand vertices inside the object; its
        threshold gates full depth, matching the existing threshold convention.
        """
        if "obj_trimesh" not in batch:
            raise ValueError("SDF guidance requires batch['obj_trimesh']: a list of B object-local meshes")
        # Interaction gradients only update translations. Avoid backpropagating
        # through MANO/rotation just to discard those gradients afterwards.
        hand_points = (vertices - hand_trans.unsqueeze(-2)).detach() + hand_trans.unsqueeze(-2)
        points_object = (hand_points - obj_trans.unsqueeze(-2)) @ obj_rot.detach()
        sdf = self._guidance_sdf_cache.sample(
            batch["obj_trimesh"], points_object,
            pitch=cfg.get("sdf_pitch", 0.002),
            padding_voxels=cfg.get("sdf_padding_voxels", 8),
        )
        penetration = sdf.new_zeros(())
        contact = sdf.new_zeros(())
        if cfg.get("penetration_scale", 0.0) > 0:
            depth = (-sdf).clamp_min(0)
            penetration = (
                depth * (depth > cfg.get("penetration_threshold", 0.0))
            ).mean(dim=-1).sum()
        if cfg.get("contact_scale", 0.0) > 0:
            mask = batch["contact"].to(device=sdf.device, dtype=torch.bool)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.shape != sdf.shape:
                raise ValueError("contact must have shape [B, T, 778] (or [T, 778] for B=1)")
            contact = ((sdf.square() * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)).sum()
        loss = cfg.get("penetration_scale", 0.0) * penetration + cfg.get("contact_scale", 0.0) * contact
        return loss, penetration, contact

    def _compute_guidance(self, x_t, t_scalar, batch, guidance_cfg):
        """Compute the guidance gradient ∇_{x_t} L_guidance.

        Decodes x_t → hand mesh + object pose, evaluates penetration and
        contact and smoothness losses, and backprops through the decoding
        to obtain the gradient w.r.t. x_t.

        Args:
            x_t:          current state tensor [B, seq_len, D], requires_grad
                          will be enabled temporarily.
            t_scalar:     current timestep (scalar float), unused by the
                          loss but kept for potential schedule-dependent
                          weighting.
            batch:        conditioning dict with 'mano_params_refine'. Interaction
                          guidance requires 'obj_trimesh', a list of B trimesh
                          meshes in the same local metric coordinates as obj_metric_pc.
                          Contact also requires binary 'contact' [B, T, 778],
                          or [T, 778] for B=1.
            guidance_cfg: dict with keys:
                          - interaction_target (str): "hand", "object" (default),
                            or "both"; selects translations for interaction guidance.
                          - penetration_scale (float): weight for mean hand-vertex
                            penetration depth, summed over batch/time.
                          - penetration_threshold (float): only penalize depths
                            above this value in meters (default 0); not subtracted.
                          - contact_scale (float): weight for masked squared
                            hand-vertex-to-object-SDF distance, disabled by default.
                          - silhouette_scale (float): object pose projection loss,
                            disabled by default. Requires mask [B,T,H,W] (object=1,
                            hand=255) and intrinsics [B,T,3,3] in original pixels,
                            or existing orig_focal_length/orig_center camera fields.
                            Adds trans_anchor back before projection if present.
                          - smoothness_scale (float): shared hand and object
                            velocity + acceleration penalty weight, disabled by default.
                          - smoothness_translation_weight (float): defaults to 1.0.
                          - smoothness_rotation_weight (float): defaults to 0.05,
                            applies to global rotations and local finger poses.
                          Optional batch['frame_indices'] excludes padding triplets.
                          - sdf_pitch (float): voxel size in meters, default 0.002.
                          - sdf_padding_voxels (int): default 8, minimum 2.

        Returns:
            grad: gradient tensor, same shape as x_t, detached.
        """
        pen_scale     = guidance_cfg.get("penetration_scale", 0.0)
        contact_scale = guidance_cfg.get("contact_scale", 0.0)
        smoothness_scale = guidance_cfg.get("smoothness_scale", 0.0)
        silhouette_scale = guidance_cfg.get("silhouette_scale", 0.0)
        verbose = guidance_cfg.get("verbose", True)

        with torch.enable_grad():
            x = x_t.detach().requires_grad_(True)

            # --- If hand_only / object_only, reconstruct the full state ---
            if self.hand_only:
                full_x = torch.cat([x, batch["_x0_other_part"]], dim=2)
            elif self.object_only:
                full_x = torch.cat([batch["_x0_other_part"], x], dim=2)
            else:
                full_x = x

            betas = batch["mano_params_refine"]["betas"]
            _, pred_verts, hand_rot, hand_pose, hand_trans = self.to_mano(full_x, betas)
            obj_rot, obj_trans = self.to_object_pose(full_x)

            loss = torch.tensor(0.0, device=x.device)
            pen_loss_val = 0.0
            contact_loss_val = 0.0
            smoothness_loss = hand_trans.new_zeros(())
            smoothness_val = 0.0
            silhouette = hand_trans.new_zeros(())
            silhouette_val = 0.0

            if silhouette_scale > 0 and not self.hand_only:
                silhouette = silhouette_loss(obj_rot, obj_trans, batch, guidance_cfg)
                if verbose:
                    silhouette_val = silhouette.item()
                silhouette = silhouette_scale * silhouette

            if pen_scale > 0 or contact_scale > 0:
                loss, pen_loss, contact_loss = self._sdf_interaction_losses(
                    pred_verts, hand_trans, obj_rot, obj_trans, batch, guidance_cfg,
                )
                if verbose:
                    pen_loss_val = pen_loss.item()
                    contact_loss_val = contact_loss.item()

            # ---- Smoothness loss ----
            if smoothness_scale > 0:
                smoothness_kwargs = {
                    "translation_weight": guidance_cfg.get("smoothness_translation_weight", 1.0),
                    "rotation_weight": guidance_cfg.get("smoothness_rotation_weight", 0.05),
                    "frame_indices": batch.get("frame_indices"),
                }
                hand_smoothness = self.pose_smoothness_loss(
                    hand_trans, hand_rot.squeeze(2), hand_pose=hand_pose,
                    **smoothness_kwargs,
                )
                object_smoothness = self.pose_smoothness_loss(
                    obj_trans, obj_rot, **smoothness_kwargs,
                )
                smoothness_loss = hand_smoothness + object_smoothness
                if verbose:
                    smoothness_val = smoothness_loss.item()
                smoothness_loss = smoothness_scale * smoothness_loss

            if verbose and (
                pen_scale > 0 or contact_scale > 0 or smoothness_scale > 0 or silhouette_scale > 0
            ):
                print(
                    f"[Guidance t={float(t_scalar):.4f}] "
                    f"Pen: raw={pen_loss_val:.6f}, scaled={pen_loss_val * pen_scale:.6f} | "
                    f"Contact: raw={contact_loss_val:.6f}, scaled={contact_loss_val * contact_scale:.6f} | "
                    f"Smoothness: raw={smoothness_val:.6f}, scaled={smoothness_loss.item():.6f} | "
                    f"Silhouette: raw={silhouette_val:.6f}, scaled={silhouette_val * silhouette_scale:.6f}"
                )
            return self._guidance_gradients(
                x, loss, smoothness_loss,
                interaction_target=guidance_cfg.get("interaction_target", "object"),
                silhouette_loss=silhouette,
            )

    def _make_guidance_fn(self, batch, guidance_cfg):
        """Return a closure suitable for the guided samplers in BridgeMatching."""
        batch.pop("_silhouette_constraints", None)  # Observations are fixed for this sampling run.
        def _guidance_fn(x_t, t_scalar, model_kwargs):
            return self._compute_guidance(x_t, t_scalar, model_kwargs, guidance_cfg)
        return _guidance_fn

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, batch: Dict):
        """Refine poses with flow sampling or guidance directly on the input.

        guidance_cfg.guidance_only (default False) skips flow sampling and
        Gaussian initialization. It applies guidance_iters updates (falling back
        to num_guidance_steps, then 1) at t=1, using guidance_scale and
        grad_clip_norm. Flow num_steps and guidance_start_t do not apply.
        """
        batch_size, seq_len = batch["img"].shape[0], batch["img"].shape[1]
        mano_params_refine = batch["mano_params_refine"]
        use_guidance = self.guidance_cfg is not None
        guidance_only = use_guidance and self.guidance_cfg.get("guidance_only", False)

        x_0 = self.create_x0(batch)
        refine_keypoints_3d, refine_vertices, _, _, _ = self.to_mano(x_0, mano_params_refine["betas"])
        batch["refine_vertices"] = refine_vertices
        batch["refine_joints"] = refine_keypoints_3d

        if self.gaussian and not guidance_only:
            x_0 = torch.randn_like(x_0)
        if self.hand_only:
            x_0_copy = x_0.clone()
            x_0 = x_0[:, :, :self.hand_dim]
            batch["_x0_other_part"] = x_0_copy[:, :, self.hand_dim:]
        elif self.object_only:
            x_0_copy = x_0.clone()
            x_0 = x_0[:, :, self.hand_dim:]
            batch["_x0_other_part"] = x_0_copy[:, :, :self.hand_dim]

        if use_guidance:
            guidance_fn = self._make_guidance_fn(batch, self.guidance_cfg)
            g_scale   = self.guidance_cfg.get("guidance_scale", 0.01)
            g_clip    = self.guidance_cfg.get("grad_clip_norm", 1.0)
            g_steps   = self.guidance_cfg.get("num_steps", 50)
            g_start_t = self.guidance_cfg.get("guidance_start_t", 0.0)
            g_iters   = self.guidance_cfg.get("guidance_iters", self.guidance_cfg.get("num_guidance_steps", 1))

            if guidance_only:
                x = x_0.clone()
                for _ in range(g_iters):
                    grad = guidance_fn(x, 1.0, batch)
                    if g_clip is not None:
                        grad_norm = grad.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-8)
                        grad_norm = grad_norm.view(-1, *([1] * (grad.dim() - 1)))
                        grad = grad * (g_clip / grad_norm).clamp(max=1.0)
                    x = x - g_scale * grad
                traj = [x]
            elif self.bright_matching.num_steps is None:
                traj = self.bright_matching.sample_ode_guided(
                    x_0, self.denoiser, guidance_fn,
                    guidance_scale=g_scale, num_steps=g_steps,
                    grad_clip_norm=g_clip, guidance_start_t=g_start_t,
                    guidance_iters=g_iters, model_kwargs=batch,
                )
            else:
                traj = self.bright_matching.sample_few_steps_guided(
                    x_0, self.denoiser, guidance_fn,
                    guidance_scale=g_scale, grad_clip_norm=g_clip,
                    guidance_start_t=g_start_t, guidance_iters=g_iters,
                    model_kwargs=batch,
                    guide_final_prediction=(
                        self.guidance_cfg.get("smoothness_scale", 0.0) > 0
                        or self.guidance_cfg.get("silhouette_scale", 0.0) > 0
                    ),
                )
        else:
            if self.bright_matching.num_steps is None:
                traj = self.bright_matching.sample_ode(x_0, self.denoiser, self.inference_steps, model_kwargs=batch)
            else:
                traj = self.bright_matching.sample_few_steps(x_0, self.denoiser, model_kwargs=batch)

        out = traj[-1]
        self.denoiser.clear_cache()

        if self.hand_only:
            out = torch.cat([out, x_0_copy[:, :, self.hand_dim:]], dim=2)
        elif self.object_only:
            out = torch.cat([x_0_copy[:, :, :self.hand_dim], out], dim=2)

        pred_keypoints_3d, pred_vertices, pred_global_orient, pred_hand_pose, pred_hand_trans = self.to_mano(out, mano_params_refine["betas"])

        predictions = {}
        predictions["pred_vertices"] = pred_vertices # (B, seq_len, 778, 3)
        predictions["pred_keypoints_3d"] = pred_keypoints_3d # (B, seq_len, 21, 3)
        predictions["pred_global_orient"] = pred_global_orient # (B, seq_len, 1, 3, 3)
        predictions["pred_hand_pose"] = pred_hand_pose # (B, seq_len, 15, 3, 3)
        predictions["pred_hand_trans"] = pred_hand_trans # (B, seq_len, 3)
        predictions["pred_betas"] = mano_params_refine["betas"] # (B, seq_len, 10)

        pred_obj_pc, objRot, objTrans = self.to_object(out, batch["obj_metric_pc"])
        predictions["pred_obj_pc"] = pred_obj_pc
        predictions["pred_obj_rot"] = objRot # (B, seq_len, 3, 3)
        predictions["pred_obj_trans"] = objTrans # (B, seq_len, 3)

        return predictions

    def create_x0(self, batch: Dict):
        batch_size, seq_len = batch["img"].shape[0], batch["img"].shape[1]

        mano_params_refine = batch["mano_params_refine"]
        global_orient_refine = mano_params_refine["global_orient"].reshape(batch_size, seq_len, 1, 3, 3)
        hand_pose_refine = mano_params_refine["hand_pose"].reshape(batch_size, seq_len, 15, 3, 3)
        # betas_refine = mano_params_refine["betas"]
        trans_refine = mano_params_refine["trans"].reshape(batch_size, seq_len, 3)

        global_orient_rhand_6d = global_orient_refine.transpose(-1, -2)[..., :2, :].reshape(batch_size, seq_len, 6)
        hand_pose_refine_6d = hand_pose_refine.transpose(-1, -2)[..., :2, :].reshape(batch_size, seq_len, 15 * 6)

        objRot_refine = axis_angle_to_matrix(batch['obj_rot_refine'].reshape(batch_size, seq_len, 3))
        objRot_refine_6d = objRot_refine.transpose(-1, -2)[..., :2, :].reshape(batch_size, seq_len, 6)
        objTrans_refine = batch['obj_trans_refine'].reshape(batch_size, seq_len, 3)

        if self.trans_delta == "h2o":
            objTrans_refine = objTrans_refine - trans_refine
        elif self.trans_delta == "o2h":
            trans_refine = trans_refine - objTrans_refine

        x_0_hand = torch.cat([global_orient_rhand_6d, hand_pose_refine_6d, trans_refine], dim=2) # (B, seq_len, 6 + 15*6 + 3)
        x_0_obj = torch.cat([objRot_refine_6d, objTrans_refine], dim=2) # (B, seq_len, 6 + 3)

        x_0 = torch.cat([x_0_hand, x_0_obj], dim=2) # (B, seq_len, 108)

        return x_0


    def to_mano(self, x: torch.Tensor, betas: torch.Tensor):
        hand_x = x[:, :, :self.hand_dim] # (B, seq_len, 6 + 15*6 + 3)
        batch_size, seq_len = hand_x.shape[0], hand_x.shape[1]

        global_orient_rhand_6d = hand_x[:, :, :6]
        global_orient_rhand_rotmat = rotation_6d_to_matrix(global_orient_rhand_6d.reshape(batch_size * seq_len, 6)).reshape(batch_size * seq_len, 1, 3, 3)
        fpose_rhand_6d = hand_x[:, :, 6:6+15*6]
        fpose_rhand_rotmat = rotation_6d_to_matrix(fpose_rhand_6d.reshape(batch_size * seq_len, 15, 6)).reshape(batch_size * seq_len, 15, 3, 3)
        trans_hand = hand_x[:, :, 6+15*6:6+15*6+3]

        if self.trans_delta == "o2h":
            obj_x = x[:, :, self.hand_dim:]
            objTrans = obj_x[:, :, 6:6+3]
            trans_hand = trans_hand + objTrans

        hand_params = {}
        hand_params['global_orient'] = global_orient_rhand_rotmat
        hand_params['hand_pose'] = fpose_rhand_rotmat

        # if betas doesn't have seq_len dim, we expand it
        if betas.dim() == 2:
            betas = betas.unsqueeze(1).expand(-1, seq_len, -1)
        hand_params['betas'] = betas.reshape(batch_size * seq_len, -1)

        mano_output = self.denoiser.mano(**{k: v.float() for k,v in hand_params.items()}, pose2rot=False)
        mano_joints = mano_output.joints.reshape(batch_size, seq_len, 21, 3) # (B, seq_len, 21, 3)
        mano_vertices = mano_output.vertices.reshape(batch_size, seq_len, 778, 3) # (B, seq_len, 778, 3)
        mano_joints += trans_hand.unsqueeze(2) # (B, seq_len, 1, 3)
        mano_vertices += trans_hand.unsqueeze(2)

        global_orient_rhand_rotmat = global_orient_rhand_rotmat.reshape(batch_size, seq_len, 1, 3, 3)
        fpose_rhand_rotmat = fpose_rhand_rotmat.reshape(batch_size, seq_len, 15, 3, 3)

        return mano_joints, mano_vertices, global_orient_rhand_rotmat, fpose_rhand_rotmat, trans_hand

    def to_object_pose(self, x: torch.Tensor):
        """Decode the object pose without transforming its sampled point cloud."""
        obj_x = x[:, :, self.hand_dim:] # (B, seq_len, 6 + 3)

        objRot_6d = obj_x[:, :, :6]
        objRot = rotation_6d_to_matrix(objRot_6d) # (B, seq_len, 3, 3)
        objTrans = obj_x[:, :, 6:6+3] # (B, seq_len, 3)

        if self.trans_delta == "h2o":
            hand_x = x[:, :, :self.hand_dim]
            trans_hand = hand_x[:, :, -3:]  # Both full-pose and translation-only hands.
            objTrans = objTrans + trans_hand

        return objRot, objTrans

    def to_object(self, x: torch.Tensor, obj_metric_pc: torch.Tensor):
        objRot, objTrans = self.to_object_pose(x)
        pred_obj_pc = torch.matmul(obj_metric_pc, objRot.transpose(-1, -2)) + objTrans.unsqueeze(-2)
        return pred_obj_pc, objRot, objTrans


    def forward(self, batch: Dict):
        if self.training:
            raise RuntimeError("This release supports inference only; call model.eval() first.")
        return self.generate(batch)
