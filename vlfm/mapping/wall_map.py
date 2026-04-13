# vlfm/mapping/wall_map.py
# Copyright ...
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
import hashlib

try:
    import open3d as o3d
except Exception:
    o3d = None  # clustering disabled without open3d

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import get_point_cloud, transform_points
try:
    from vlfm.utils.img_utils import fill_small_holes
except Exception:
    fill_small_holes = None


class WallMap(BaseMap):
    """Wall-only point cloud and top-down occupancy map (separate from ObstacleMap).
    Supports DBSCAN instance clustering and visualization."""

    def __init__(
        self,
        size: int = 1000,
        pixels_per_meter: int = 20,
        hole_area_thresh: int = 100000,
    ) -> None:
        super().__init__(size, pixels_per_meter)
        self._occ = np.zeros((size, size), dtype=bool)      # wall occupancy (top-down)
        self._cloud = np.empty((0, 3), dtype=np.float32)    # accumulated global 3D wall cloud
        self._hole_area_thresh = int(hole_area_thresh)
        from vlfm.config import load_config
        self._wm_cfg = load_config().get("wall_map", {})

    # ---------- Lifecycle ----------
    def reset(self) -> None:
        super().reset()
        self._occ.fill(0)
        self._cloud = np.empty((0, 3), dtype=np.float32)

    # ---------- Queries & visualization ----------
    def get_cloud(self) -> np.ndarray:
        return self._cloud.copy()

    def get_occupancy(self) -> np.ndarray:
        return self._occ.copy()

    def visualize(self) -> np.ndarray:
        """Visualize wall occupancy as a top-down image."""
        vis = np.ones((*self._occ.shape, 3), np.uint8) * 255
        vis[self._occ] = (200, 60, 60)
        vis = cv2.flip(vis, 0)
        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(vis, self._camera_positions, self._last_camera_yaw)
        return vis

    def _filter_wall_cloud(
        self, pts: np.ndarray,
        min_range: float, max_range: float,
        stat_nb: int, stat_std: float,
        rad_nb: int, rad_radius: float,
        ransac_dist: float, ransac_iter: int,
        max_planes: int, vertical_nz_max: float, min_inliers: int,
    ) -> np.ndarray:
        """Post-process 3D wall cloud:
        1) XY distance gate
        2) Statistical/radius outlier removal
        3) RANSAC vertical plane extraction (|nz| <= vertical_nz_max)"""
        if pts.shape[0] == 0:
            return pts
        # (1) Range gate
        d_xy = np.linalg.norm(pts[:, :2], axis=1)
        mask = (d_xy >= float(min_range)) & (d_xy <= float(max_range))
        z_pre_min = 0.8
        z_max = 3
        mask &= (pts[:, 2] >= z_pre_min) & (pts[:, 2] <= z_max)
        pts = pts[mask]

        if pts.shape[0] < 50 or o3d is None:
            return pts

        # (2) Downsample + outlier removal (reduce point count before RANSAC)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        # Voxel downsample (default 5cm; 0 or negative to disable)
        voxel = self._wm_cfg.get("voxel_m", 0.05)
        if voxel and voxel > 0:
            try:
                pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
            except Exception:
                pass

        # Optional uniform/random downsample (stable time bound)
        cap_n = self._wm_cfg.get("sample_cap_n", 0)
        if cap_n and len(pcd.points) > cap_n:
            prob = float(cap_n) / float(len(pcd.points))
            try:
                pcd = pcd.random_down_sample(prob)
            except Exception:
                # fallback: every-k
                k = max(1, int(round(len(pcd.points) / float(cap_n))))
                pcd = pcd.uniform_down_sample(every_k_points=k)
        try:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=int(stat_nb), std_ratio=float(stat_std))
        except Exception:
            pass
        try:
            pcd, _ = pcd.remove_radius_outlier(nb_points=int(rad_nb), radius=float(rad_radius))
        except Exception:
            pass

        # (3) Multi-plane extraction (keep only vertical planes)
        kept = []
        rem = pcd
        z_min = 1
        z_frac = 0.6
        for _ in range(int(max_planes)):
            if len(rem.points) < int(min_inliers):
                break
            model, inliers = rem.segment_plane(
                distance_threshold=float(ransac_dist),
                ransac_n=3,
                num_iterations=int(ransac_iter),
            )
            a, b, c, _ = model  # plane normal (a, b, c)
            # Keep only vertical planes (small |nz| along height axis)
            if abs(float(c)) <= float(vertical_nz_max):
                pts_in = np.asarray(rem.select_by_index(inliers).points)
                if pts_in.shape[0] > 0:
                    # Height fraction check for wall classification
                    hi = pts_in[:, 2] >= z_min
                    if float(np.mean(hi)) >= z_frac:
                        pts_hi = pts_in[hi]
                        kept.append(pts_hi)
            rem = rem.select_by_index(inliers, invert=True)
        if len(kept) > 0:
            return np.vstack(kept)
        # No vertical planes found; return outlier-filtered points
        return np.asarray(rem.points)

    def update_from_depth(
        self,
        depth: np.ndarray,                 # [0,1] normalized
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> None:
        """Extract wall planes from depth via Open3D RANSAC (uses only depth < max_depth)."""
        if depth is None or depth.size == 0:
            return
        # 1) Depth hole correction and scale to meters
        if self._hole_area_thresh != -1 and fill_small_holes is not None:
            filled_depth = fill_small_holes(depth, self._hole_area_thresh)
        else:
            filled_depth = depth.copy()
            filled_depth[filled_depth == 0] = 1.0
        scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
        # 2) max_depth mask
        depth_valid = (scaled_depth > 0) & (scaled_depth < max_depth)
        if not np.any(depth_valid):
            return
        # 3) Camera-frame point cloud -> episodic coords
        mask_u8 = depth_valid.astype(np.uint8)
        cloud_cam = get_point_cloud(scaled_depth, mask_u8, fx, fy)
        if cloud_cam.shape[0] == 0:
            return
        cloud_epi = transform_points(tf_camera_to_episodic, cloud_cam)
        # 4) Keep only RANSAC vertical planes
        _wm = self._wm_cfg
        cloud_epi = self._filter_wall_cloud(
            cloud_epi,
            min_range=_wm["min_range_m"],
            max_range=_wm["depth_max_range_m"],
            stat_nb=_wm["stat_nb"],
            stat_std=_wm["stat_std"],
            rad_nb=_wm["rad_nb"],
            rad_radius=_wm["rad_radius"],
            ransac_dist=_wm["ransac_dist"],
            ransac_iter=_wm["ransac_iters"],
            max_planes=_wm["max_planes"],
            vertical_nz_max=_wm["vertical_nz_max"],
            min_inliers=_wm["min_inliers"],
        )
        if cloud_epi.shape[0] == 0:
            return
        # 5) Accumulate and update top-down occupancy
        if self._cloud.shape[0] == 0:
            self._cloud = cloud_epi.astype(np.float32)
        else:
            self._cloud = np.vstack([self._cloud, cloud_epi.astype(np.float32)])
        xy = cloud_epi[:, :2]
        px = self._xy_to_px(xy)
        h, w = self._occ.shape
        valid = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        px = px[valid]
        if px.shape[0] == 0:
            return
        self._occ[px[:, 1], px[:, 0]] = True


    def _color_for_category(self, name: str) -> tuple[int, int, int]:
        """Deterministic hash-based BGR color for a category name (range 50-230)."""
        h = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(h & 0xFFFFFFFF)
        b, g, r = [int(x) for x in rng.randint(50, 230, size=3)]
        return (b, g, r)  # OpenCV BGR

    def _draw_points_bgr(self, vis: np.ndarray, pts_xy: np.ndarray, color_bgr: tuple[int,int,int]) -> None:
        """Project xy (meters) to pixels and paint them with color_bgr."""
        if pts_xy.size == 0:
            return
        px = self._xy_to_px(pts_xy)  # (N,2) int
        h, w = vis.shape[:2]
        m = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        px = px[m]
        if px.shape[0] == 0:
            return
        vis[px[:, 1], px[:, 0]] = color_bgr

    def _draw_center_bgr(self, vis: np.ndarray, center_xy: np.ndarray, color_bgr: tuple[int,int,int]) -> None:
        """Draw instance center as a small circle (black border + fill color)."""
        p = self._xy_to_px(center_xy.reshape(1, 2))[0]
        x, y = int(p[0]), int(p[1])
        h, w = vis.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(vis, (x, y), 3, (0, 0, 0), -1)
            cv2.circle(vis, (x, y), 2, color_bgr, -1)

    def visualize_with_objects(
        self,
        object_clouds: Dict[str, List[np.ndarray]],
        draw_centers: bool = True,
        draw_legend: bool = True,
        prefer_within_range: bool = True,
        explored_mask: Optional[np.ndarray] = None,
        wall_color: tuple[int, int, int] = (64, 64, 64),
        wall_thickness_px: int = 3,
        explored_color: tuple[int, int, int] = (230, 230, 230),
    ) -> np.ndarray:
        """Visualize walls + objects + (optionally) explored area together."""
        # 1) White background
        vis = np.ones((*self._occ.shape, 3), np.uint8) * 255

        # 2) (Optional) Shade explored area in light gray
        if explored_mask is not None:
            m = explored_mask.astype(bool)
            # Resize if dimensions differ
            if m.shape != self._occ.shape:
                m = cv2.resize(m.astype(np.uint8), (self._occ.shape[1], self._occ.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            vis[m] = explored_color

        # 3) Walls: thick dark gray
        occ_u8 = self._occ.astype(np.uint8)
        if wall_thickness_px > 1:
            k = np.ones((wall_thickness_px, wall_thickness_px), np.uint8)
            occ_draw = cv2.dilate(occ_u8, k, iterations=1).astype(bool)
        else:
            occ_draw = occ_u8.astype(bool)
        vis[occ_draw] = wall_color

        # 4) Object point clouds (per-category colors)
        if object_clouds:
            for cat, inst_list in object_clouds.items():
                color = self._color_for_category(cat)
                if not isinstance(inst_list, list):
                    continue
                for inst in inst_list:
                    if not isinstance(inst, np.ndarray) or inst.size == 0:
                        continue
                    pts = inst
                    if prefer_within_range and inst.shape[1] >= 4:
                        wmask = (inst[:, -1] == 1)
                        if np.any(wmask):
                            pts = inst[wmask]
                    self._draw_points_bgr(vis, pts[:, :2], color)
                    if draw_centers and pts.shape[0] > 0:
                        cxy = np.mean(pts[:, :2], axis=0).astype(np.float32)
                        self._draw_center_bgr(vis, cxy, color)

        # 5) Flip vertically and overlay trajectory
        vis = cv2.flip(vis, 0)
        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(vis, self._camera_positions, self._last_camera_yaw)

        # 6) (Optional) Legend
        if draw_legend and object_clouds:
            x0, y0, dy = 10, 10, 18
            for i, cat in enumerate(sorted(object_clouds.keys())):
                color = self._color_for_category(cat)
                y = y0 + i * dy
                cv2.rectangle(vis, (x0, y), (x0 + 14, y + 14), color, -1)
                cv2.putText(vis, cat[:18], (x0 + 20, y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
        return vis