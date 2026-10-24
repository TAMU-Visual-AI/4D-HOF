"""HO3D metrics from the research evaluator (inputs in meters)."""

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

# Close the MANO wrist for the containment test used by the original evaluator.
WRIST_FACES = np.array([
    [92, 38, 122], [234, 92, 122], [239, 234, 122], [279, 239, 122],
    [215, 279, 122], [215, 122, 118], [215, 118, 117], [215, 117, 119],
    [215, 119, 120], [215, 120, 108], [215, 108, 79], [215, 79, 78],
    [215, 78, 121], [214, 215, 121],
])


def chamfer_f_scores(source, target):
    """Symmetric sum of mean squared distances in cm²; F at 5/10 mm."""
    source, target = source * 100, target * 100
    d1 = cKDTree(source).query(target)[0]
    d2 = cKDTree(target).query(source)[0]
    scores = []
    for threshold in (0.5, 1.0):
        p1 = np.float32(np.mean(d1 < threshold))
        p2 = np.float32(np.mean(d2 < threshold))
        scores.append(2 * p1 * p2 / (p1 + p2 + 1e-7))
    return np.mean(d1 ** 2) + np.mean(d2 ** 2), *scores


def hand_relative_metrics(pred_mesh, pred_joints, pred_pose, gt_mesh, gt_joints, gt_pose,
                          *, generator):
    """Wrist-relative object CD (cm²) and hand MPJPE (mm) over frames."""
    frame_cd = []
    for i in range(len(pred_pose)):
        pred = trimesh.transform_points(pred_mesh.vertices, pred_pose[i]) - pred_joints[i, 0]
        gt = trimesh.transform_points(gt_mesh.vertices, gt_pose[i]) - gt_joints[i, 0]
        pi = torch.randperm(len(pred), generator=generator)[:3000].numpy()
        gi = torch.randperm(len(gt), generator=generator)[:3000].numpy()
        cd, _, _ = chamfer_f_scores(pred[pi], gt[gi])
        frame_cd.append(float(cd))
    metrics = {"cd_hand": np.mean(frame_cd)}
    relative_pred = pred_joints - pred_joints[:, :1]
    relative_gt = gt_joints - gt_joints[:, :1]
    metrics["mpjpe_mm"] = np.linalg.norm(relative_pred - relative_gt, axis=-1).mean() * 1000

    return {key: float(value) for key, value in metrics.items()}, frame_cd


def physical_metrics(pred_mesh, hand_vertices, hand_faces, object_pose, contact):
    """Original 5 mm surface-voxel intersection proxy and binary contact accuracy."""
    pitch = 0.005
    points = pred_mesh.voxelized(pitch=pitch).points  # Deliberately not filled.
    volumes = []
    for vertices, pose in zip(hand_vertices, object_pose):
        hand = trimesh.Trimesh(vertices=vertices, faces=hand_faces)
        inside = hand.contains(trimesh.transform_points(points, pose))
        volumes.append(inside.sum() * (pitch * 100) ** 3)
    volumes = np.asarray(volumes, dtype=np.float32)
    contact = np.asarray(contact, dtype=bool)
    if volumes.shape != contact.shape:
        raise ValueError("Predicted and ground-truth contact lengths differ")
    return {"intersection_volume_cm3": float(volumes.mean()),
            "contact_accuracy": float(np.mean((volumes != 0) == contact))}


def motion_jerk(joints, frames):
    """Mean third difference at 6 fps in m/s³; never bridge missing frames."""
    runs = np.split(np.arange(len(frames)), np.flatnonzero(np.diff(frames) != 5) + 1)
    norms = [np.linalg.norm(np.diff(joints[run], n=3, axis=0), axis=-1)
             for run in runs if len(run) >= 4]
    return float(np.concatenate(norms).mean() * 6 ** 3) if norms else 0.0
