"""Chunked HO3D flow matching inference."""

import numpy as np
import torch
from tqdm import tqdm

from hof.utils.transform import matrix_to_axis_angle_np, axis_angle_to_matrix_np
from .windows import inference_stride, inference_windows, WindowPredictionBlender

@torch.no_grad()
def infer_chunk(model, batch):
    """Center a model chunk like VideoRefineDataset, then restore camera space."""
    if not model.normalize_trans:
        return model(batch)

    hand_trans = batch["mano_params_refine"]["trans"]
    batch_size = hand_trans.shape[0]
    anchor = hand_trans.reshape(batch_size, -1, 3).mean(dim=1)
    # Copy mappings and use out-of-place shifts to preserve the caller's inputs.
    centered = dict(batch)
    centered["trans_anchor"] = anchor
    centered["mano_params_refine"] = dict(batch["mano_params_refine"])
    centered["mano_params_refine"]["trans"] = hand_trans - anchor.reshape(
        batch_size, *([1] * (hand_trans.ndim - 2)), 3
    )
    centered["obj_trans_refine"] = batch["obj_trans_refine"] - anchor[:, None, :]
    centered["obj_pc_refine"] = batch["obj_pc_refine"] - anchor[:, None, None, :]

    output = model(centered)
    for key in ("pred_hand_trans", "pred_obj_trans"):
        output[key] = output[key] + anchor[:, None, :]
    for key in ("pred_vertices", "pred_keypoints_3d", "pred_obj_pc"):
        output[key] = output[key] + anchor[:, None, None, :]
    return output
def to_device(batch, device):
    for key, value in batch.items():
        if isinstance(value, dict):
            to_device(value, device)
        elif isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def infer_sequence(model, data, input_cfg, *, chunk_size=128, stride_ratio=0.5,
                   previous=None, description="HO3D"):
    """Refine one sequence, optionally starting from a previous pass's output."""
    from .images import preprocess_image

    inference_stride(chunk_size, stride_ratio)
    count = len(data["image_paths"])
    if previous is None:
        hand = data["hand"]
        obj_rot = matrix_to_axis_angle_np(data["pose"][:, :3, :3])
        obj_trans = data["pose"][:, :3, 3]
    else:
        hand = {key: previous[pred] for key, pred in {
            "global_orient": "pred_global_orient", "hand_pose": "pred_hand_pose",
            "betas": "pred_betas", "trans": "pred_hand_trans",
        }.items()}
        obj_rot = matrix_to_axis_angle_np(previous["pred_obj_rot"])
        obj_trans = previous["pred_obj_trans"]

    vertices = np.asarray(data["mesh"].vertices)
    point_count = input_cfg.get("obj_sample_num", 4096)
    indices = np.random.choice(len(vertices), size=point_count, replace=len(vertices) < point_count)
    local_points = vertices[indices].astype(np.float32)
    windows = inference_windows(0, count, chunk_size, stride_ratio)
    blender = WindowPredictionBlender(0, count, chunk_size, stride_ratio)

    def tensor(value):
        return torch.from_numpy(np.asarray(value)).unsqueeze(0).float()

    for indices, valid_start in tqdm(windows, desc=description):
        # Preprocess only this window to keep full-video RGB memory bounded.
        frames = np.stack([
            preprocess_image(str(data["image_paths"][i]),
                             img_size=input_cfg.get("img_size", 224), mask=data["mask"][i],
                             apply_mask=input_cfg.get("apply_mask", False),
                             full_image=input_cfg.get("full_image", True),
                             rescale_factor=input_cfg.get("rescale_factor", 1.0))
            for i in indices
        ])
        rots = (data["pose"][indices, :3, :3] if previous is None
                else axis_angle_to_matrix_np(obj_rot[indices]))
        points = local_points @ rots.transpose(0, 2, 1) + obj_trans[indices, None, :]
        batch = {
            "mask": data["mask"][indices][None],
            "frame_indices": torch.tensor(indices, dtype=torch.long)[None],
            "img": tensor(frames),
            "contact": torch.from_numpy(data["contact"][indices])[None].bool(),
            "mano_params_refine": {key: tensor(value[indices]) for key, value in hand.items()},
            "obj_rot_refine": tensor(obj_rot[indices]),
            "obj_trans_refine": tensor(obj_trans[indices]),
            "obj_pc_refine": tensor(points),
            "obj_metric_pc": tensor(np.tile(local_points, (len(indices), 1, 1))),
            "obj_trimesh": [data["mesh"]],
            "orig_center": tensor(data["orig_center"][indices]),
            "orig_focal_length": tensor(data["orig_focal_length"][indices]),
        }
        output = infer_chunk(model, to_device(batch, model.device))
        prediction = {key: output[key][0].detach().cpu().numpy() for key in blender.keys}
        if any(not np.isfinite(value).all() for value in prediction.values()):
            raise RuntimeError(f"Non-finite predictions in {description}, frame {indices[0]}")
        blender.add(indices, prediction, valid_start)
    result = blender.finalize()
    result["pred_hand_trans"] = result["pred_hand_trans"].reshape(-1, 1, 3)
    return result
