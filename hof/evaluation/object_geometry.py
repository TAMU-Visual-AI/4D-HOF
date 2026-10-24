"""Object-only geometry evaluation from mv.py's scale_ho3d_v3raw path."""

import copy
import warnings

import numpy as np
import trimesh
from tqdm import tqdm

from .metrics import chamfer_f_scores


def _features(cloud, o3d):
    voxel_size = 0.005
    down = cloud.voxel_down_sample(voxel_size)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 2, max_nn=30))
    features = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return down, features


def aligned_object_scores(target_mesh, source_mesh, *, num_iters=1200, progress=True):
    """Center Open3D mesh copies, fit similarity ICP, and score all vertices.

    The target is ground truth. Registration uses 1000 surface points per mesh,
    5 mm FPFH voxels, and the original RANSAC/ICP thresholds and stopping rule.
    """
    import open3d as o3d

    target_mesh = copy.deepcopy(target_mesh)
    source_mesh = copy.deepcopy(source_mesh)
    source_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(source_mesh.vertices) - source_mesh.get_center())
    target_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(target_mesh.vertices) - target_mesh.get_center())
    target = target_mesh.sample_points_uniformly(1000)
    source = source_mesh.sample_points_uniformly(1000)
    source_down, source_features = _features(source, o3d)
    target_down, target_features = _features(target, o3d)
    registration = o3d.pipelines.registration

    def refine(initial, threshold):
        return registration.registration_icp(
            source, target, threshold, initial,
            registration.TransformationEstimationPointToPoint(with_scaling=True))

    def score(transform):
        aligned = copy.deepcopy(source_mesh).transform(transform)
        return chamfer_f_scores(np.asarray(aligned.vertices), np.asarray(target_mesh.vertices))

    initial = refine(np.eye(4), 0.01)
    best = score(initial.transformation)
    for iteration in tqdm(range(num_iters), desc="Object geometry alignment", disable=not progress):
        threshold = 0.02 if iteration > num_iters / 2 else 0.1
        try:
            coarse = registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_features, target_features, True, threshold,
                registration.TransformationEstimationPointToPoint(False), 3,
                [registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                 registration.CorrespondenceCheckerBasedOnDistance(threshold)],
                registration.RANSACConvergenceCriteria(100000, 0.999))
            refined = refine(coarse.transformation, threshold)
        except RuntimeError as error:
            warnings.warn(f"Skipping failed object registration: {error}", RuntimeWarning)
            continue
        candidate = score(refined.transformation)
        if candidate[0] < best[0]:
            best = candidate
        if best[2] > 0.99:
            break
    return tuple(float(value) for value in best)


def object_geometry_metrics(pred_mesh, gt_mesh, *, seed, progress=True):
    """One CD/F5/F10 per sequence, independent of hand and predicted poses."""
    import open3d as o3d

    o3d.utility.random.seed(seed % (2 ** 31))
    # Same starting convention as eval_sequence_v3_raw; never mutate inference meshes.
    pred_mesh = pred_mesh.copy()
    pred_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]))

    def convert(mesh):
        result = o3d.geometry.TriangleMesh()
        result.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        result.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
        return result

    cd, f5, f10 = aligned_object_scores(convert(gt_mesh), convert(pred_mesh), progress=progress)
    return {"cd": cd, "f5": f5, "f10": f10}
