"""Unified ObjectPointCloudMap -- main class that composes the three sub-modules.

The ``enable_depth_reprojection`` constructor flag controls whether the
depth-to-RGB reprojection block (COIN variant) is active.  When disabled
(PSL variant) the raw ``object_mask`` is used directly.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from vlfm.utils.geometry_utils import (
    extract_yaw,
    transform_points,
    within_fov_cone,
)

from vlfm.mapping._cloud_geometry import (
    compute_aligned_mask,
    extract_object_cloud,
    get_closest_point,
    open3d_dbscan_filtering,
    too_offset,
)
from vlfm.mapping._instance_management import _Instance, InstanceManagementMixin
from vlfm.mapping._blacklist_and_scoring import BlacklistAndScoringMixin


class ObjectPointCloudMap(InstanceManagementMixin, BlacklistAndScoringMixin):
    """Maintains per-class 3-D point clouds with instance tracking and scoring."""

    clouds: Dict[str, np.ndarray] = {}
    use_dbscan: bool = True

    def __init__(
        self,
        erosion_size: float,
        enable_depth_reprojection: bool = True,
    ) -> None:
        self._erosion_size = erosion_size
        self._enable_depth_reprojection = enable_depth_reprojection

        self.last_target_coord: Union[np.ndarray, None] = None

        # Scoring tables
        self._candidate_scores: Dict[str, Dict[str, Tuple[np.ndarray, float]]] = defaultdict(dict)
        self._candidate_ctx_scores: Dict[str, Dict[str, Tuple[np.ndarray, float]]] = defaultdict(dict)
        self._candidate_ctxcount_scores: Dict[str, Dict[str, Tuple[np.ndarray, float]]] = defaultdict(dict)
        self._candidate_rel_scores: Dict[str, Dict[str, Tuple[np.ndarray, float]]] = defaultdict(dict)
        self._promoted_relctx_sums: Dict[str, dict] = defaultdict(dict)

        # Instance tracking
        self._instances: Dict[str, List[_Instance]] = defaultdict(list)
        from vlfm.config import load_config
        _oc_cfg = load_config().get("object_cloud", {})
        self.instance_eps_m: float = _oc_cfg.get("instance_eps_m", 0.35)
        self.instance_min_pts: int = _oc_cfg.get("instance_min_pts", 20)
        self._updates: int = 0

        # Blacklist / failed
        self._blacklist: Dict[str, List[Tuple[np.ndarray, float]]] = defaultdict(list)
        self._blacklist_instances: Dict[str, List[np.ndarray]] = defaultdict(list)

        # Detection buffer
        self.detection_cloud: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.clouds: Dict[str, List[np.ndarray]] = defaultdict(list)
        self._latest_detection_idx: Dict[str, int] = {}
        self._detection_frame_idx: Dict[str, List[int]] = defaultdict(list)
        self._promoted_match_for_detection: Dict[str, List[int]] = defaultdict(list)

        # Caches
        self._vox_cache: dict = {}
        self._cache_cap = _oc_cfg.get("det_cache_cap", 512)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.clouds = defaultdict(list)
        self.detection_cloud = defaultdict(list)
        self.last_target_coord = None

        self._blacklist = defaultdict(list)
        self._blacklist_instances = defaultdict(list)

        self._candidate_scores.clear()
        self._candidate_ctx_scores = defaultdict(dict)
        self._candidate_ctxcount_scores = defaultdict(dict)
        self._candidate_rel_scores = defaultdict(dict)
        self._promoted_relctx_sums = defaultdict(dict)

        self._instances.clear()
        self._updates = 0

        self._latest_detection_idx = {}
        self._detection_frame_idx = defaultdict(list)
        self._promoted_match_for_detection = defaultdict(list)

        self._vox_cache = {}

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def has_object(self, target_class: str) -> bool:
        insts = self.clouds.get(target_class, [])
        if not insts:
            return False
        return any(isinstance(c, np.ndarray) and c.size > 0 for c in insts)

    def update_map(
        self,
        object_name: str,
        depth_img: np.ndarray,
        object_mask: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        is_target_object: bool,
        frame_id: Optional[int] = None,
    ) -> Tuple[bool, int, int]:
        """Update the object map with a new detection.

        Returns:
            (created_new, detection_idx, matched_promoted_idx)
        """
        self._updates += 1

        # Depth-to-RGB reprojection (COIN) or passthrough (PSL)
        if self._enable_depth_reprojection:
            aligned_mask = compute_aligned_mask(
                depth_img, object_mask, fx, fy, min_depth, max_depth,
            )
        else:
            aligned_mask = object_mask

        # Skip far-sight dummy clouds
        try:
            md_th = 0.985
            md_ratio_th = 0.98
            masked = depth_img[aligned_mask > 0]
            if masked.size > 0 and float(np.mean(masked >= md_th)) >= md_ratio_th:
                return (False, -1, -1)
        except Exception:
            pass

        local_cloud = extract_object_cloud(
            depth_img, aligned_mask, min_depth, max_depth, fx, fy, self._erosion_size,
        )
        if len(local_cloud) == 0:
            return (False, -1, -1)

        # Mark offset / out-of-range detections with random IDs
        if too_offset(aligned_mask):
            within_range = np.ones_like(local_cloud[:, 0]) * np.random.rand()
        else:
            within_range = (local_cloud[:, 0] <= max_depth * 0.95) * 1.0
            within_range = within_range.astype(np.float32)
            within_range[within_range == 0] = np.random.rand()

        global_cloud = transform_points(tf_camera_to_episodic, local_cloud)
        global_cloud = np.concatenate((global_cloud, within_range[:, None]), axis=1)

        # Blacklist filter
        if len(global_cloud) > 0 and len(self._blacklist.get(object_name, [])) > 0:
            mask_valid = np.ones((global_cloud.shape[0],), dtype=bool)
            xy = global_cloud[:, :2]
            for center_xy, radius in self._blacklist[object_name]:
                dist = np.linalg.norm(xy - center_xy[None, :], axis=1)
                mask_valid &= (dist >= radius)
            global_cloud = global_cloud[mask_valid]
            if len(global_cloud) == 0:
                return (False, -1, -1)

        # Too-close check
        curr_position = tf_camera_to_episodic[:3, 3]
        closest_point = get_closest_point(global_cloud, curr_position, use_dbscan=self.use_dbscan)
        dist = np.linalg.norm(closest_point[:3] - curr_position)
        if dist < 1.0:
            return (False, -1, -1)

        if self._matches_any_blacklisted_instance(object_name, global_cloud):
            return (False, -1, -1)

        # Assign to detection buffer
        idx, matched_prom_idx, created_new = self._assign_detection_instance(
            object_name, global_cloud, frame_id=frame_id,
        )
        self._latest_detection_idx[object_name] = idx

        return (bool(created_new), int(idx), int(matched_prom_idx))

    def commit_latest_detection_to_clouds(self, object_name: str, set_last_coord: bool = True) -> bool:
        """Promote the latest detection instance for the class (backward compat)."""
        if object_name not in self._latest_detection_idx:
            return False
        idx = int(self._latest_detection_idx[object_name])
        return self.commit_detection_instance_to_clouds(object_name, idx, set_last_coord=set_last_coord)

    def get_closest_point(self, cloud: np.ndarray, curr_position: np.ndarray) -> np.ndarray:
        """Return the cloud point nearest to *curr_position*."""
        return get_closest_point(cloud, curr_position, use_dbscan=self.use_dbscan)

    def get_best_object(self, target_class: str, curr_position: np.ndarray) -> Union[np.ndarray, None]:
        """Choose nearest non-blacklisted promoted instance centre; smooth updates."""
        insts = self.clouds.get(target_class, [])
        if len(insts) == 0:
            target_cloud = self.get_target_cloud(target_class)
            if target_cloud.shape[0] == 0:
                return None
            closest_point_2d = get_closest_point(
                target_cloud, curr_position, use_dbscan=self.use_dbscan
            )[:2]
        else:
            centers = np.vstack([self._cloud_center_xy(inst) for inst in insts])
            dists = np.linalg.norm(centers - np.asarray(curr_position[:2]).reshape(1, 2), axis=1)
            closest_point_2d = get_closest_point(
                insts[int(np.argmin(dists))], curr_position, use_dbscan=self.use_dbscan
            )[:2]

        if self.last_target_coord is None:
            self.last_target_coord = closest_point_2d
        else:
            delta_dist = np.linalg.norm(closest_point_2d - self.last_target_coord)
            if delta_dist < 0.1:
                return self.last_target_coord
            elif delta_dist < 0.5 and np.linalg.norm(curr_position[:2] - closest_point_2d) > 2.0:
                return self.last_target_coord
            else:
                self.last_target_coord = closest_point_2d
        return self.last_target_coord

    def update_explored(
        self, tf_camera_to_episodic: np.ndarray, max_depth: float, cone_fov: float
    ) -> None:
        """Remove out-of-range points that are now within the camera's FOV cone."""
        camera_coordinates = tf_camera_to_episodic[:3, 3]
        camera_yaw = extract_yaw(tf_camera_to_episodic)

        for obj, insts in list(self.clouds.items()):
            kept: List[np.ndarray] = []
            for inst in insts:
                if inst.shape[0] == 0:
                    continue
                within_range = within_fov_cone(
                    camera_coordinates, camera_yaw, cone_fov, max_depth * 0.5, inst,
                )
                range_ids = set(within_range[..., -1].tolist())
                for range_id in range_ids:
                    if range_id == 1:
                        continue
                    inst = inst[inst[..., -1] != range_id]
                if inst.shape[0] > 0:
                    kept.append(inst)
            self.clouds[obj] = kept
            self._rebuild_instances_from_clouds(obj)

    # ------------------------------------------------------------------
    # Target cloud helpers
    # ------------------------------------------------------------------

    def get_target_cloud(self, target_class: str) -> np.ndarray:
        """Aggregated promoted cloud; prefers within-range points."""
        insts = self.clouds.get(target_class, [])
        if not insts:
            return np.empty((0, 4), dtype=np.float32)
        parts = [c for c in insts if isinstance(c, np.ndarray) and c.size > 0]
        stacked = np.concatenate(parts, axis=0) if parts else np.empty((0, 4), np.float32)
        if stacked.shape[0] > 0 and np.any(stacked[:, -1] == 1):
            stacked = stacked[stacked[:, -1] == 1]
        return stacked

    def nearest_candidate_center(self, object_name: str, coord_xy: np.ndarray) -> np.ndarray:
        centers = self.list_candidate_centers(object_name)
        if len(centers) == 0:
            cloud = self.get_target_cloud(object_name)
            if cloud.shape[0] == 0:
                return np.asarray(coord_xy, dtype=np.float32).reshape(2)
            return np.mean(cloud[:, :2], axis=0).astype(np.float32)
        centers_arr = np.vstack([np.asarray(c, dtype=np.float32).reshape(2) for c in centers])
        dists = np.linalg.norm(
            centers_arr - np.asarray(coord_xy, dtype=np.float32).reshape(1, 2), axis=1
        )
        return centers_arr[int(np.argmin(dists))]
