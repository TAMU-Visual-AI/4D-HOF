"""Load released predictions and raw HO3D train annotations for evaluation."""

import pickle
from pathlib import Path

import numpy as np
import torch
import trimesh

from hof.utils.transform import axis_angle_to_matrix_np, matrix_to_axis_angle_np
from .object_geometry import object_geometry_metrics
from .metrics import WRIST_FACES, hand_relative_metrics, motion_jerk, physical_metrics

JOINT_MAP = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
CV_FLIP = np.array([1, -1, -1], dtype=np.float32)


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def create_mano(model_path):
    from hof.model.mano_wrapper import MANO

    if not (model_path / "MANO_RIGHT.pkl").is_file():
        raise FileNotFoundError(f"Place MANO_RIGHT.pkl at {model_path / 'MANO_RIGHT.pkl'}")
    return MANO(model_path=str(model_path), num_hand_joints=15).eval()


def load_predictions(path, expected_frames):
    data = load_pickle(path)
    shapes = {"pred_global_orient": (1, 3, 3), "pred_hand_pose": (15, 3, 3),
              "pred_betas": (10,), "pred_hand_trans": (1, 3),
              "pred_obj_rot": (3, 3), "pred_obj_trans": (3,)}
    result = {}
    count = None
    for key, shape in shapes.items():
        value = np.asarray(data[key])
        if key == "pred_hand_trans" and value.ndim == 2:
            value = value[:, None, :]
        if value.ndim != len(shape) + 1 or value.shape[1:] != shape:
            raise ValueError(f"{path}: {key} must have shape (T, {shape}), got {value.shape}")
        if count is None:
            count = len(value)
        if len(value) != count or len(value) < expected_frames:
            raise ValueError(f"{path}: inconsistent or incomplete {key}; need {expected_frames} frames")
        value = value[:expected_frames]
        if not np.isfinite(value).all():
            raise ValueError(f"{path}: {key} contains non-finite values")
        result[key] = value
    return result


def load_ground_truth(ho3d_root, sequence, protocol_frames, valid_frames):
    meta_dir = ho3d_root / "train" / sequence / "meta"
    meta_files = sorted(meta_dir.glob("*.pkl"))[:1024]
    if not meta_files:
        raise FileNotFoundError(f"No HO3D annotations in {meta_dir}")
    # Predictions follow the original zero-based HO3D frame order.
    if [int(path.stem) for path in meta_files] != list(range(len(meta_files))):
        raise ValueError(f"{meta_dir}: expected consecutive frame IDs starting at 0000")
    frames = [frame for frame in protocol_frames if frame < len(meta_files)
              and f"{sequence}/{meta_files[frame].stem}" in valid_frames]
    if not frames:
        raise ValueError(f"{sequence}: no valid evaluation frames")
    joints, poses, contacts = [], [], []
    object_name = None
    for frame in frames:
        meta = load_pickle(meta_files[frame])
        if object_name is not None and object_name != meta["objName"]:
            raise ValueError(f"{sequence}: object identity changes across frames")
        object_name = meta["objName"]
        joint = np.asarray(meta["handJoints3D"])
        if joint.shape != (21, 3):
            raise ValueError(f"{meta_files[frame]}: expected 21 annotated hand joints")
        joints.append(joint[JOINT_MAP] * CV_FLIP)
        rot = axis_angle_to_matrix_np(np.asarray(meta["objRot"]).reshape(3))
        # Keep the original evaluator's matrix -> axis-angle -> matrix round trip.
        rot = axis_angle_to_matrix_np(matrix_to_axis_angle_np(np.diag(CV_FLIP) @ rot))
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = np.asarray(meta["objTrans"]).reshape(3) * CV_FLIP
        poses.append(pose)
        if "handVertContact" not in meta:
            raise KeyError(f"{meta_files[frame]}: handVertContact annotations are required")
        contacts.append(bool(np.any(meta["handVertContact"])))
    joints, poses = np.stack(joints), np.stack(poses)
    if not np.isfinite(joints).all() or not np.isfinite(poses).all():
        raise ValueError(f"{sequence}: non-finite ground-truth geometry")
    mesh = trimesh.load(ho3d_root / "models" / object_name / "textured_simple.obj")
    return frames, joints, poses, contacts, mesh, len(meta_files)


@torch.no_grad()
def evaluate_sequence(sequence, *, prediction_dir, init_root, ho3d_root,
                      protocol_frames, valid_frames, mano, seed):
    frames, gt_joints, gt_pose, contact, gt_mesh, count = load_ground_truth(
        ho3d_root, sequence, protocol_frames, valid_frames)
    predictions = load_predictions(prediction_dir / f"{sequence}.pkl", count)
    pred_mesh = load_pickle(init_root / sequence / "mesh.pkl")["mesh"]
    for name, mesh in (("predicted", pred_mesh), ("ground-truth", gt_mesh)):
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
            raise ValueError(f"{sequence}: {name} object must be a nonempty triangle mesh")
        if not np.isfinite(mesh.vertices).all():
            raise ValueError(f"{sequence}: non-finite {name} mesh vertices")
    hand = mano(global_orient=torch.as_tensor(predictions["pred_global_orient"][frames]).float(),
                hand_pose=torch.as_tensor(predictions["pred_hand_pose"][frames]).float(),
                betas=torch.as_tensor(predictions["pred_betas"][frames]).float(), pose2rot=False)
    trans = predictions["pred_hand_trans"][frames]
    pred_joints = hand.joints.numpy() + trans
    pred_vertices = hand.vertices.numpy() + trans
    pred_pose = np.tile(np.eye(4), (len(frames), 1, 1))
    pred_pose[:, :3, :3] = predictions["pred_obj_rot"][frames]
    pred_pose[:, :3, 3] = predictions["pred_obj_trans"][frames]
    metrics, _ = hand_relative_metrics(
        pred_mesh, pred_joints, pred_pose, gt_mesh, gt_joints, gt_pose,
        generator=torch.Generator().manual_seed(seed))
    metrics.update(object_geometry_metrics(pred_mesh, gt_mesh, seed=seed))
    # Trimesh retries ambiguous containment rays using NumPy's global RNG.
    # Isolate its seed so a sequence has the same result in a subset or full run.
    random_state = np.random.get_state()
    try:
        np.random.seed(seed % (2 ** 32))
        metrics.update(physical_metrics(pred_mesh, pred_vertices,
                                       np.concatenate([mano.faces, WRIST_FACES]), pred_pose, contact))
    finally:
        np.random.set_state(random_state)
    metrics["jerk_m_s3"] = motion_jerk(pred_joints, frames)
    return {"num_frames": len(frames), "metrics": metrics}
