"""Sampled, occlusion-aware object silhouette loss for flow guidance."""

import numpy as np
import torch
import torch.nn.functional as F


def prepare_silhouette(batch, cfg):
    """Prepare fixed mask constraints once per batch/chunk, in original image space.

    mask: [B, T, H, W], labels 0=background, 1=object, 255=hand.
    intrinsics: optional [B, T, 3, 3] pixel camera matrices; otherwise use
    orig_focal_length [B, T] and orig_center [B, T, 2] (fx=fy).
    """
    import cv2
    from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
    import trimesh

    masks = torch.as_tensor(batch["mask"]).detach().cpu().numpy()
    B, T, source_h, source_w = masks.shape
    factor = cfg.get("silhouette_downsample", 4)
    h, w = max(8, round(source_h / factor)), max(8, round(source_w / factor))
    boundary_count = cfg.get("silhouette_boundary_samples", 128)
    distance_fields = np.zeros((B * T, h, w), dtype=np.float32)
    boundary = np.zeros((B * T, boundary_count, 2), dtype=np.float32)
    boundary_valid = np.zeros((B * T, boundary_count), dtype=bool)
    active = np.zeros(B * T, dtype=bool)
    dilation = cfg.get("silhouette_occluder_dilation", 2)
    for i, mask in enumerate(masks.reshape(B * T, source_h, source_w)):
        reduced = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        obj, hand = reduced == 1, reduced == 255
        if dilation > 0:
            hand = binary_dilation(hand, iterations=dilation)
        distance_fields[i] = distance_transform_edt(~(obj | hand))
        ys, xs = np.nonzero(obj & ~binary_erosion(obj))
        if len(xs) < 3:
            continue
        if len(xs) > boundary_count:
            selected = np.linspace(0, len(xs) - 1, boundary_count).round().astype(int)
            xs, ys = xs[selected], ys[selected]
        boundary[i, :len(xs)] = np.column_stack((xs, ys))
        boundary_valid[i, :len(xs)] = True
        active[i] = True

    points = np.stack([
        trimesh.sample.sample_surface(mesh, cfg.get("silhouette_surface_samples", 768), seed=73)[0]
        for mesh in batch["obj_trimesh"]
    ]).astype(np.float32)
    if "intrinsics" in batch:
        camera = torch.as_tensor(batch["intrinsics"]).detach().float().clone()
    else:
        focal = torch.as_tensor(batch["orig_focal_length"]).detach().float().reshape(B, T)
        center = torch.as_tensor(batch["orig_center"]).detach().to(focal)
        camera = focal.new_zeros(B, T, 3, 3)
        camera[..., 0, 0] = camera[..., 1, 1] = focal
        camera[..., :2, 2] = center
        camera[..., 2, 2] = 1
    camera[..., 0, :] *= w / source_w
    camera[..., 1, :] *= h / source_h
    return {
        "points": torch.from_numpy(points),
        "camera": camera.reshape(B * T, 3, 3),
        "distance_fields": torch.from_numpy(distance_fields),
        "boundary": torch.from_numpy(boundary),
        "boundary_valid": torch.from_numpy(boundary_valid),
        "active": torch.from_numpy(active),
    }


def silhouette_loss(rotation, translation, batch, cfg):
    """Image-diagonal-normalized penalties, averaged per frame and summed over B/T.

    Cache belongs to a fixed batch/chunk. Build a new batch (or remove
    _silhouette_constraints) after changing masks, meshes, cameras, or sampling
    settings. Only the object pose and optional camera-space anchor vary per step.
    """
    if "_silhouette_constraints" not in batch:
        batch["_silhouette_constraints"] = prepare_silhouette(batch, cfg)
    data = batch["_silhouette_constraints"]
    for key, value in data.items():
        data[key] = value.to(device=rotation.device)

    if "trans_anchor" in batch:
        translation = translation + batch["trans_anchor"].to(translation)[:, None, :]
    points = data["points"][:, None] @ rotation.transpose(-1, -2) + translation.unsqueeze(-2)
    points = points.flatten(0, 1)
    camera = data["camera"]
    depth = points[..., 2]
    projected = points[..., :2] / depth.clamp_min(1e-4).unsqueeze(-1)
    projected = projected * torch.stack((camera[:, 0, 0], camera[:, 1, 1]), -1)[:, None]
    projected = projected + camera[:, None, :2, 2]
    u, v = projected.unbind(-1)
    h, w = data["distance_fields"].shape[-2:]
    grid = 2 * projected / projected.new_tensor([w - 1, h - 1]) - 1
    outside = F.grid_sample(
        data["distance_fields"][:, None], grid[:, :, None],
        mode="bilinear", padding_mode="border", align_corners=True,
    )[:, 0, :, 0]
    overflow = F.relu(-u) + F.relu(u - (w - 1)) + F.relu(-v) + F.relu(v - (h - 1))
    projected_valid = depth > 1e-4
    image_diagonal = (h * h + w * w) ** 0.5

    def reduce(values, valid):
        penalty = F.smooth_l1_loss(values, torch.zeros_like(values), reduction="none")
        return ((penalty * valid).sum(-1) / valid.sum(-1).clamp_min(1)).sum()

    containment = reduce((outside + overflow) / image_diagonal, projected_valid & data["active"][:, None])
    distances = torch.cdist(data["boundary"], projected)
    distances = distances.masked_fill(~projected_valid[:, None], 1e3)
    coverage = reduce(distances.min(-1).values / image_diagonal, data["boundary_valid"] & data["active"][:, None])
    return (containment + coverage) / 2
