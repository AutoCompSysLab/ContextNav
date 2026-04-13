import math
from typing import Tuple

import numpy as np


def rho_theta(curr_pos: np.ndarray, curr_heading: float, curr_goal: np.ndarray) -> Tuple[float, float]:
    """Calculates polar coordinates (rho, theta) relative to a given position and
    heading to a given goal position. 'rho' is the distance from the agent to the goal,
    and theta is how many radians the agent must turn (to the left, CCW from above) to
    face the goal. Coordinates are in (x, y), where x is the distance forward/backwards,
    and y is the distance to the left or right (right is negative)

    Args:
        curr_pos (np.ndarray): Array of shape (2,) representing the current position.
        curr_heading (float): The current heading, in radians. It represents how many
            radians  the agent must turn to the left (CCW from above) from its initial
            heading to reach its current heading.
        curr_goal (np.ndarray): Array of shape (2,) representing the goal position.

    Returns:
        Tuple[float, float]: A tuple of floats representing the polar coordinates
            (rho, theta).
    """
    rotation_matrix = get_rotation_matrix(-curr_heading, ndims=2)
    local_goal = curr_goal - curr_pos
    local_goal = rotation_matrix @ local_goal

    rho = np.linalg.norm(local_goal)
    theta = np.arctan2(local_goal[1], local_goal[0])

    return float(rho), float(theta)


def get_rotation_matrix(angle: float, ndims: int = 2) -> np.ndarray:
    """Returns a 2x2 or 3x3 rotation matrix for a given angle; if 3x3, the z-axis is
    rotated."""
    if ndims == 2:
        return np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
    elif ndims == 3:
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
    else:
        raise ValueError("ndims must be 2 or 3")


def wrap_heading(theta: float) -> float:
    """Wraps given angle to be between -pi and pi.

    Args:
        theta (float): The angle in radians.
    Returns:
        float: The wrapped angle in radians.
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi



def within_fov_cone(
    cone_origin: np.ndarray,
    cone_angle: float,
    cone_fov: float,
    cone_range: float,
    points: np.ndarray,
) -> np.ndarray:
    """Checks if points are within a cone of a given origin, angle, fov, and range.

    Args:
        cone_origin (np.ndarray): The origin of the cone.
        cone_angle (float): The angle of the cone in radians.
        cone_fov (float): The field of view of the cone in radians.
        cone_range (float): The range of the cone.
        points (np.ndarray): The points to check.

    Returns:
        np.ndarray: The subarray of points that are within the cone.
    """
    directions = points[:, :3] - cone_origin
    dists = np.linalg.norm(directions, axis=1)
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    angle_diffs = np.mod(angles - cone_angle + np.pi, 2 * np.pi) - np.pi

    mask = np.logical_and(dists <= cone_range, np.abs(angle_diffs) <= cone_fov / 2)
    return points[mask]


def convert_to_global_frame(agent_pos: np.ndarray, agent_yaw: float, local_pos: np.ndarray) -> np.ndarray:
    """Converts a given position from the agent's local frame to the global frame.

    Args:
        agent_pos (np.ndarray): A 3D vector representing the agent's position in their
            local frame.
        agent_yaw (float): The agent's yaw in radians.
        local_pos (np.ndarray): A 3D vector representing the position to be converted in
            the agent's local frame.

    Returns:
        A 3D numpy array representing the position in the global frame.
    """
    # Append a homogeneous coordinate of 1 to the local position vector
    local_pos_homogeneous = np.append(local_pos, 1)

    # Construct the homogeneous transformation matrix
    transformation_matrix = xyz_yaw_to_tf_matrix(agent_pos, agent_yaw)

    # Perform the transformation using matrix multiplication
    global_pos_homogeneous = transformation_matrix.dot(local_pos_homogeneous)
    global_pos_homogeneous = global_pos_homogeneous[:3] / global_pos_homogeneous[-1]

    return global_pos_homogeneous


def extract_yaw(matrix: np.ndarray) -> float:
    """Extract the yaw angle from a 4x4 transformation matrix.

    Args:
        matrix (np.ndarray): A 4x4 transformation matrix.
    Returns:
        float: The yaw angle in radians.
    """
    assert matrix.shape == (4, 4), "The input matrix must be 4x4"
    rotation_matrix = matrix[:3, :3]

    # Compute the yaw angle
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return yaw


def xyz_yaw_to_tf_matrix(xyz: np.ndarray, yaw: float) -> np.ndarray:
    """Converts a given position and yaw angle to a 4x4 transformation matrix.

    Args:
        xyz (np.ndarray): A 3D vector representing the position.
        yaw (float): The yaw angle in radians.
    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    x, y, z = xyz
    transformation_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0, x],
            [np.sin(yaw), np.cos(yaw), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )
    return transformation_matrix


def closest_point_within_threshold(points_array: np.ndarray, target_point: np.ndarray, threshold: float) -> int:
    """Find the point within the threshold distance that is closest to the target_point.

    Args:
        points_array (np.ndarray): An array of 2D points, where each point is a tuple
            (x, y).
        target_point (np.ndarray): The target 2D point (x, y).
        threshold (float): The maximum distance threshold.

    Returns:
        int: The index of the closest point within the threshold distance.
    """
    distances = np.sqrt((points_array[:, 0] - target_point[0]) ** 2 + (points_array[:, 1] - target_point[1]) ** 2)
    within_threshold = distances <= threshold

    if np.any(within_threshold):
        closest_index = np.argmin(distances)
        return int(closest_index)

    return -1


def transform_points(transformation_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    # Add a homogeneous coordinate of 1 to each point for matrix multiplication
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))

    # Apply the transformation matrix to the points
    transformed_points = np.dot(transformation_matrix, homogeneous_points.T).T

    # Remove the added homogeneous coordinate and divide by the last coordinate
    return transformed_points[:, :3] / transformed_points[:, 3:]


def get_point_cloud(depth_image: np.ndarray, mask: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Calculates the 3D coordinates (x, y, z) of points in the depth image based on
    the horizontal field of view (HFOV), the image width and height, the depth values,
    and the pixel x and y coordinates.

    Args:
        depth_image (np.ndarray): 2D depth image.
        mask (np.ndarray): 2D binary mask identifying relevant pixels.
        fx (float): Focal length in the x direction.
        fy (float): Focal length in the y direction.

    Returns:
        np.ndarray: Array of 3D coordinates (x, y, z) of the points in the image plane.
    """
    v, u = np.where(mask)
    z = depth_image[v, u]
    x = (u - depth_image.shape[1] // 2) * z / fx
    y = (v - depth_image.shape[0] // 2) * z / fy
    cloud = np.stack((z, -x, -y), axis=-1)

    return cloud


def get_fov(focal_length: float, image_height_or_width: int) -> float:
    """
    Given an fx and the image width, or an fy and the image height, returns the
    horizontal or vertical field of view, respectively.

    Args:
        focal_length: Focal length of the camera.
        image_height_or_width: Height or width of the image.

    Returns:
        Field of view in radians.
    """

    # Calculate the field of view using the formula
    fov = 2 * math.atan((image_height_or_width / 2) / focal_length)
    return fov


def pt_from_rho_theta(rho: float, theta: float) -> np.ndarray:
    """
    Given a rho and theta, computes the x and y coordinates.

    Args:
        rho: Distance from the origin.
        theta: Angle from the x-axis.

    Returns:
        numpy.ndarray: x and y coordinates.
    """
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)

    return np.array([x, y])


def intrinsics_from_hfov(hfov_deg: float, width: int, height: int):
    """Compute camera intrinsics (fx, fy, cx, cy, K) from HFOV and image size."""
    hfov = math.radians(float(hfov_deg))
    fx = float(width) / (2.0 * math.tan(hfov / 2.0))
    # fy derived from VFOV: tan(vfov/2) = (H/W) * tan(hfov/2)
    vfov = 2.0 * math.atan((float(height) / float(width)) * math.tan(hfov / 2.0))
    fy = float(height) / (2.0 * math.tan(vfov / 2.0))
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return fx, fy, cx, cy, K


def depth_mask_via_reprojection(
    depth_norm: np.ndarray,
    rgb_mask: np.ndarray,
    K_depth: np.ndarray,
    K_rgb: np.ndarray,
    R_d2r: np.ndarray = None,
    t_d2r: np.ndarray = None,
    min_depth: float = 0.0,
    max_depth: float = 5.0,
) -> np.ndarray:
    """Reproject each depth pixel into the RGB frame and return a uint8 mask
    indicating which depth pixels land inside rgb_mask (>0)."""
    H, W = depth_norm.shape[:2]
    # Treat zero-depth holes as far
    d = depth_norm.copy()
    d[d == 0] = 1.0
    z = d * (float(max_depth) - float(min_depth)) + float(min_depth)  # meters

    # Pixel grid
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    fx_d, fy_d, cx_d, cy_d = float(K_depth[0, 0]), float(K_depth[1, 1]), float(K_depth[0, 2]), float(K_depth[1, 2])

    X = (uu - cx_d) * z / max(1e-6, fx_d)
    Y = (vv - cy_d) * z / max(1e-6, fy_d)

    pts_d = np.stack([X, Y, z], axis=-1).reshape(-1, 3)  # (N,3)

    # Depth-to-RGB extrinsic parameters
    if R_d2r is None:
        R_d2r = np.eye(3, dtype=np.float32)
    if t_d2r is None:
        t_d2r = np.zeros(3, dtype=np.float32)

    pts_r = (pts_d @ R_d2r.T) + t_d2r.reshape(1, 3)
    Xr, Yr, Zr = pts_r[:, 0], pts_r[:, 1], pts_r[:, 2]
    valid = Zr > 1e-6

    fx_r, fy_r, cx_r, cy_r = float(K_rgb[0, 0]), float(K_rgb[1, 1]), float(K_rgb[0, 2]), float(K_rgb[1, 2])
    ur = fx_r * (Xr / Zr) + cx_r
    vr = fy_r * (Yr / Zr) + cy_r

    ur_i = np.round(ur).astype(np.int32)
    vr_i = np.round(vr).astype(np.int32)

    H2, W2 = rgb_mask.shape[:2]
    inside = (ur_i >= 0) & (ur_i < W2) & (vr_i >= 0) & (vr_i < H2)
    idx = np.where(valid & inside)[0]

    rgb_hit = np.zeros(uu.size, dtype=np.uint8)
    if idx.size > 0:
        rgb_hit[idx] = (rgb_mask[vr_i[idx], ur_i[idx]] > 0).astype(np.uint8)

    return rgb_hit.reshape(H, W)