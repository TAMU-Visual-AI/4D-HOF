"""Read prepared HO3D initialization; no reconstruction pipelines are required."""

from pathlib import Path
import pickle

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_pickle(path):
    # Initialization bundles contain NumPy arrays and trimesh objects.
    # Only load bundles from a trusted source.
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _array(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: contains non-finite values")
    return array.astype(np.float32, copy=False)


def load_sequence(init_root, ho3d_root, sequence):
    """Load the first 1024 train frames (or all frames of a shorter sequence).

    Intrinsics are normalized by the original image width/height, matching the
    research initialization. Bundles may contain the full sequence or its first
    1024 frames. The mean focal length is computed from the stored intrinsics.
    """
    import trimesh

    folder = Path(init_root) / sequence
    rgb_folder = Path(ho3d_root) / "train" / sequence / "rgb"
    image_paths = sorted(p for p in rgb_folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    total = len(image_paths)
    if total == 0:
        raise ValueError(f"No RGB frames found in {rgb_folder}")
    count = min(total, 1024)

    masks = np.asarray(read_pickle(folder / "mask.pkl")["mask"])
    if masks.ndim != 3 or masks.shape[0] not in {count, total}:
        raise ValueError(f"mask: expected ({count}, H, W) or ({total}, H, W), got {masks.shape}")
    stored_frames = masks.shape[0]
    if any(not np.isin(frame, [0, 1, 255]).all() for frame in masks):
        raise ValueError("mask labels must be 0=background, 1=object, 255=hand")
    if masks.dtype != np.uint8:
        masks = masks.astype(np.uint8)
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if first is None or first.shape[:2] != masks.shape[1:]:
        raise ValueError("RGB resolution must match the original-resolution masks")

    contact = np.asarray(read_pickle(folder / "contact.pkl")["contact"])
    if contact.shape != (stored_frames, 778) or not np.isin(contact, [0, 1]).all():
        raise ValueError(f"contact: expected binary ({stored_frames}, 778), got {contact.shape}")

    camera_path = folder / "intrinsics.pkl"
    if not camera_path.is_file():
        camera_path = folder / "depth.pkl"
    camera_data = read_pickle(camera_path)
    intrinsics = _array(camera_data["intrinsics"], (stored_frames, 3, 3), "intrinsics").copy()
    del camera_data
    height, width = masks.shape[1:]
    intrinsics[:, 0, :] *= width
    intrinsics[:, 1, :] *= height
    if np.any(intrinsics[:, [0, 1], [0, 1]] <= 0):
        raise ValueError("Camera focal lengths must be positive")
    focal = np.mean((intrinsics[:, 0, 0] + intrinsics[:, 1, 1]) / 2)
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("Mean focal length must be positive and finite")

    mesh = read_pickle(folder / "mesh.pkl")["mesh"]
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError("mesh must be a nonempty trimesh.Trimesh with faces")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("mesh vertices must be finite, in object-local meters")
    pose = _array(read_pickle(folder / "pose.pkl")["pose"], (stored_frames, 4, 4), "pose")
    hand = read_pickle(folder / "hand.pkl")
    shapes = {"global_orient": (stored_frames, 1, 3, 3), "hand_pose": (stored_frames, 15, 3, 3),
              "betas": (stored_frames, 10), "trans": (stored_frames, 1, 3)}
    hand = {key: _array(hand[key], shape, key)[:count] for key, shape in shapes.items()}
    return {
        "image_paths": image_paths[:count],
        "mask": masks[:count],
        "contact": contact[:count].astype(bool),
        "mesh": mesh,
        "pose": pose[:count],
        "hand": hand,
        "orig_center": intrinsics[:count, :2, 2].copy(),
        "orig_focal_length": np.full(count, focal, dtype=np.float32),
    }
