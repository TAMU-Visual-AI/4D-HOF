import numpy as np
from scipy.spatial.transform import Rotation

def inference_stride(chunk_size, stride_ratio):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not np.isfinite(stride_ratio) or not 0 < stride_ratio <= 1:
        raise ValueError("stride_ratio must be finite and in (0, 1]")
    stride = max(1, int(chunk_size * stride_ratio))
    if 2 * stride < chunk_size:
        raise ValueError(
            "stride_ratio must give a stride of at least ceil(chunk_size / 2) "
            "so no more than two windows overlap"
        )
    return stride


def inference_windows(start, stop, chunk_size, stride_ratio=0.5):
    """Return (frame indices, valid start) pairs, excluding padding from fusion.

    Stride equal to chunk size preserves legacy tail context and discards its
    duplicate predictions. Overlapping inference blends the entire tail window,
    shifted to end at stop. If adding it would create triple overlap, it replaces
    the preceding window instead; this retains full coverage without padding.
    """
    stride = inference_stride(chunk_size, stride_ratio)
    if not 0 <= start < stop:
        raise ValueError("inference requires 0 <= start < stop")
    length = stop - start
    if length < chunk_size:
        padding = chunk_size - length
        return [([start] * padding + list(range(start, stop)), padding)]
    if stride == chunk_size:
        windows = []
        for i in range(start, stop, stride):
            window_start = min(i, stop - chunk_size)
            windows.append((list(range(window_start, window_start + chunk_size)),
                            i - window_start))
        return windows
    starts = list(range(start, stop - chunk_size + 1, stride))
    if starts[-1] != stop - chunk_size:
        if len(starts) >= 2 and stop - chunk_size < starts[-2] + chunk_size:
            starts.pop()
        starts.append(stop - chunk_size)
    return [(list(range(i, i + chunk_size)), 0) for i in starts]


def slerp_rotations(first, second, fraction):
    """Shortest-path quaternion SLERP, batched over frames and hand joints."""
    shape = first.shape
    q1 = Rotation.from_matrix(first.reshape(-1, 3, 3)).as_quat()
    q2 = Rotation.from_matrix(second.reshape(-1, 3, 3)).as_quat()
    t = np.broadcast_to(fraction, shape[:-2]).reshape(-1, 1)
    dot = np.sum(q1 * q2, axis=-1, keepdims=True)
    # q and -q represent the same rotation; choose the shorter arc.
    q2 = np.where(dot < 0, -q2, q2)
    dot = np.clip(np.abs(dot), 0, 1)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    # Normalized linear interpolation avoids division by zero near identity.
    near = sine < 1e-8
    denominator = np.where(near, 1.0, sine)
    w1 = np.where(near, 1 - t, np.sin((1 - t) * angle) / denominator)
    w2 = np.where(near, t, np.sin(t * angle) / denominator)
    quaternion = w1 * q1 + w2 * q2
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    return Rotation.from_quat(quaternion).as_matrix().reshape(shape)


class WindowPredictionBlender:
    """Fuse same-frame estimates with positive triangular window weights.

    Two rotation estimates use SLERP with fraction w_new / (w_old + w_new).
    Frames with a single estimate retain their original rotation matrices.
    """

    keys = ("pred_global_orient", "pred_hand_pose", "pred_hand_trans",
            "pred_betas", "pred_obj_rot", "pred_obj_trans")
    rotation_keys = frozenset(("pred_global_orient", "pred_hand_pose", "pred_obj_rot"))

    def __init__(self, start, stop, chunk_size, stride_ratio=0.5):
        stride = inference_stride(chunk_size, stride_ratio)
        self.start = start
        self.weight_sum = np.zeros(stop - start, dtype=np.float64)
        self.count = np.zeros(stop - start, dtype=np.int64)
        positions = np.arange(chunk_size)
        self.weights = (np.minimum(positions + 1, chunk_size - positions)
                        if stride < chunk_size else np.ones(chunk_size))
        self.sums = {}
        self.rotations = {}
        self.dtypes = {}

    def add(self, frame_indices, predictions, valid_start=0):
        # Padded copies of the first frame must never receive extra votes.
        indices = np.asarray(frame_indices[valid_start:]) - self.start
        weights = self.weights[valid_start:]
        if np.any(self.count[indices] >= 2):
            raise ValueError("SLERP blending supports at most two predictions per frame")
        overlap = self.count[indices] == 1
        fraction = weights[overlap] / (self.weight_sum[indices[overlap]] + weights[overlap])
        self.weight_sum[indices] += weights
        self.count[indices] += 1
        for key in self.keys:
            values = predictions[key][valid_start:]
            self.dtypes[key] = values.dtype
            if key in self.rotation_keys:
                if key not in self.rotations:
                    self.rotations[key] = np.zeros(
                        (len(self.weight_sum), *values.shape[1:]), dtype=np.float64
                    )
                self.rotations[key][indices[~overlap]] = values[~overlap]
                if np.any(overlap):
                    self.rotations[key][indices[overlap]] = slerp_rotations(
                        self.rotations[key][indices[overlap]], values[overlap],
                        fraction.reshape((-1,) + (1,) * (values.ndim - 3)),
                    )
                continue
            if key not in self.sums:
                self.sums[key] = np.zeros(
                    (len(self.weight_sum), *values.shape[1:]), dtype=np.float64
                )
            self.sums[key][indices] += values * weights.reshape(
                (-1,) + (1,) * (values.ndim - 1)
            )

    def finalize(self):
        if not np.all(self.weight_sum > 0):
            raise ValueError("Some sequence frames have no predictions")
        results = {key: value.astype(self.dtypes[key])
                   for key, value in self.rotations.items()}
        for key, total in self.sums.items():
            mean = total / self.weight_sum.reshape(
                (-1,) + (1,) * (total.ndim - 1)
            )
            results[key] = mean.astype(self.dtypes[key])
        return results
