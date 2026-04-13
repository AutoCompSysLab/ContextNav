"""Point-cloud extraction, filtering, and depth-to-RGB reprojection helpers."""

from typing import Tuple

import cv2
import numpy as np
import open3d as o3d

from vlfm.utils.geometry_utils import (
    get_point_cloud,
    intrinsics_from_hfov,
    depth_mask_via_reprojection,
)


def open3d_dbscan_filtering(
    points: np.ndarray, eps: float = 0.2, min_points: int = 100
) -> np.ndarray:
    """Return the largest DBSCAN cluster from *points* (Nx3)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    labels = np.array(pcd.cluster_dbscan(eps, min_points))

    unique_labels, label_counts = np.unique(labels, return_counts=True)

    non_noise_mask = unique_labels != -1
    non_noise_labels = unique_labels[non_noise_mask]
    non_noise_counts = label_counts[non_noise_mask]

    if len(non_noise_labels) == 0:
        return np.array([])

    largest_label = non_noise_labels[np.argmax(non_noise_counts)]
    return points[labels == largest_label]


def too_offset(mask: np.ndarray) -> bool:
    """True if the mask bounding-box lies entirely in the left or right third."""
    x, y, w, h = cv2.boundingRect(mask)
    third = mask.shape[1] // 3

    if x + w <= third:
        return x <= int(0.05 * mask.shape[1])
    elif x >= 2 * third:
        return x + w >= int(0.95 * mask.shape[1])
    return False


# ---------------------------------------------------------------------------
# Depth-to-RGB reprojection (COIN-specific, gated by enable flag)
# ---------------------------------------------------------------------------

def _rotz(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], np.float32)


def _roty(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], np.float32)


def _rotx(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], np.float32)


def compute_aligned_mask(
    depth_img: np.ndarray,
    object_mask: np.ndarray,
    fx: float,
    fy: float,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    """Build a depth-aligned mask by reprojecting from depth FOV to RGB FOV.

    Returns:
        aligned_mask with same spatial shape as *depth_img*.
    """
    H, W = depth_img.shape[:2]
    K_depth = np.array(
        [[float(fx), 0.0, (W - 1) / 2.0],
         [0.0, float(fy), (H - 1) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    from vlfm.config import load_config
    _cfg = load_config()
    _vis_cfg = _cfg.get("visualization", {})
    _cg_cfg = _cfg.get("cloud_geometry", {})

    rgb_hfov_deg = _vis_cfg.get("rgb_hfov_deg", 42)
    _, _, _, _, K_rgb = intrinsics_from_hfov(rgb_hfov_deg, W, H)

    roll = np.deg2rad(_cg_cfg.get("d2r_roll_deg", 0.0))
    pitch = np.deg2rad(_cg_cfg.get("d2r_pitch_deg", 0.0))
    yaw = np.deg2rad(_cg_cfg.get("d2r_yaw_deg", 0.0))
    R_d2r = _rotz(yaw) @ _roty(pitch) @ _rotx(roll)
    t_d2r = np.array(
        [_cg_cfg.get("d2r_tx", 0.0),
         _cg_cfg.get("d2r_ty", 0.0),
         _cg_cfg.get("d2r_tz", 0.0)],
        dtype=np.float32,
    )

    aligned_mask = depth_mask_via_reprojection(
        depth_norm=depth_img,
        rgb_mask=(object_mask > 0).astype(np.uint8),
        K_depth=K_depth,
        K_rgb=K_rgb,
        R_d2r=R_d2r,
        t_d2r=t_d2r,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    return aligned_mask


# ---------------------------------------------------------------------------
# Cloud extraction and closest-point query
# ---------------------------------------------------------------------------

def extract_object_cloud(
    depth: np.ndarray,
    object_mask: np.ndarray,
    min_depth: float,
    max_depth: float,
    fx: float,
    fy: float,
    erosion_size: int,
) -> np.ndarray:
    """Extract a local 3-D point cloud for the masked object region."""
    final_mask = object_mask * 255
    final_mask = cv2.erode(final_mask, None, iterations=erosion_size)  # type: ignore

    valid_depth = depth.copy()
    valid_depth[valid_depth == 0] = 1  # holes become far
    valid_depth = valid_depth * (max_depth - min_depth) + min_depth
    return get_point_cloud(valid_depth, final_mask, fx, fy)


def get_closest_point(
    cloud: np.ndarray,
    curr_position: np.ndarray,
    use_dbscan: bool = True,
) -> np.ndarray:
    """Return the cloud point nearest to *curr_position*."""
    ndim = curr_position.shape[0]
    if use_dbscan:
        return cloud[np.argmin(np.linalg.norm(cloud[:, :ndim] - curr_position, axis=1))]

    if ndim == 2:
        ref_point = np.concatenate((curr_position, np.array([0.5])))
    else:
        ref_point = curr_position
    distances = np.linalg.norm(cloud[:, :3] - ref_point, axis=1)
    sorted_indices = np.argsort(distances)
    percent = 0.25
    top_percent = sorted_indices[: int(percent * len(cloud))]
    try:
        median_index = top_percent[int(len(top_percent) / 2)]
    except IndexError:
        median_index = 0
    return cloud[median_index]
