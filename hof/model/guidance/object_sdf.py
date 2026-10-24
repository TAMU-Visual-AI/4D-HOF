"""Cached object-local voxel SDFs for differentiable HOI guidance."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ObjectSDF:
    values: torch.Tensor  # [1, 1, Z, Y, X], negative inside
    object_to_index: torch.Tensor
    pitch: float

    @classmethod
    def from_mesh(cls, mesh, *, pitch, padding_voxels, device):
        from scipy.ndimage import distance_transform_edt

        # Catch unit mistakes before allocating a potentially enormous volume.
        shape = np.ceil(mesh.extents / pitch) + 3 + 2 * padding_voxels
        if np.prod(shape) > 32_000_000:
            raise ValueError("Object SDF is too large; check mesh units (meters) or increase sdf_pitch")
        voxels = mesh.voxelized(pitch).fill()
        occupied = np.pad(np.asarray(voxels.matrix, dtype=bool), padding_voxels)
        inside = distance_transform_edt(occupied)
        outside = distance_transform_edt(~occupied)
        values = np.where(occupied, -(inside - 0.5) * pitch, (outside - 0.5) * pitch)
        index_to_object = np.asarray(voxels.transform, dtype=np.float64).copy()
        index_to_object[:3, 3] -= index_to_object[:3, :3] @ np.full(3, padding_voxels)
        return cls(
            torch.as_tensor(
                np.ascontiguousarray(values.transpose(2, 1, 0)),
                dtype=torch.float32, device=device,
            )[None, None],
            torch.as_tensor(np.linalg.inv(index_to_object), dtype=torch.float32, device=device),
            pitch,
        )

    def sample(self, points):
        """Sample [..., 3] points, retaining an attraction gradient outside the grid."""
        points = points.float()
        indices = points @ self.object_to_index[:3, :3].T + self.object_to_index[:3, 3]
        upper = points.new_tensor(self.values.shape[-3:][::-1]) - 1
        clamped = torch.minimum(indices.clamp_min(0), upper)
        outside_distance = torch.linalg.vector_norm(indices - clamped, dim=-1) * self.pitch
        grid = 2 * clamped / upper - 1
        values = F.grid_sample(
            self.values, grid.reshape(1, -1, 1, 1, 3),
            mode="bilinear", padding_mode="border", align_corners=True,
        )
        return values.reshape(points.shape[:-1]) + outside_distance


class ObjectSDFCache:
    """Keep only the current batch's unique SDFs; reuse meshes shared with the next.

    Trimesh's content hash invalidates entries after vertex/face edits. Device,
    pitch, and padding changes also invalidate them. No autograd graphs are cached.
    """

    def __init__(self):
        self._entries = {}

    def clear(self):
        self._entries.clear()

    def sample(self, meshes, points_object, *, pitch=0.002, padding_voxels=8):
        import trimesh

        if not isinstance(meshes, (list, tuple)) or len(meshes) != points_object.shape[0]:
            raise ValueError("batch['obj_trimesh'] must be a list of B trimesh.Trimesh objects")
        if not np.isfinite(pitch) or pitch <= 0:
            raise ValueError("sdf_pitch must be positive and finite")
        if int(padding_voxels) != padding_voxels or padding_voxels < 2:
            raise ValueError("sdf_padding_voxels must be an integer >= 2")
        padding_voxels = int(padding_voxels)
        entries = {}
        samplers = []
        for mesh in meshes:
            if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
                raise ValueError("batch['obj_trimesh'] must contain nonempty trimesh.Trimesh objects")
            key = (hash(mesh), float(pitch), padding_voxels, points_object.device)
            sampler = entries.get(key, self._entries.get(key))
            if sampler is None:
                if not np.isfinite(mesh.vertices).all():
                    raise ValueError("Object mesh vertices must be finite")
                # generate() runs under no_grad; also support inference_mode callers.
                with torch.inference_mode(False), torch.no_grad():
                    sampler = ObjectSDF.from_mesh(
                        mesh, pitch=float(pitch), padding_voxels=padding_voxels,
                        device=points_object.device,
                    )
            entries[key] = sampler
            samplers.append(sampler)
        self._entries = entries
        return torch.stack([sampler.sample(points) for sampler, points in zip(samplers, points_object)])
