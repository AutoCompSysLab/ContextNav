"""Instance tracking, detection-buffer management, and promotion logic."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d


@dataclass
class _Instance:
    """Persistent, lightweight per-instance representation."""
    center_xy: np.ndarray        # shape (2,)
    n_points: int = 0            # running count for weighted updates
    last_update: int = 0         # internal update counter

    def __post_init__(self) -> None:
        self.center_xy = np.asarray(self.center_xy, dtype=np.float32).reshape(2)


# ---------------------------------------------------------------------------
# Mixin that supplies all instance-related methods to ObjectPointCloudMap
# ---------------------------------------------------------------------------

class InstanceManagementMixin:
    """Mixed into ObjectPointCloudMap to provide instance tracking."""

    # These attributes are expected on the host class (set in __init__):
    #   _instances, instance_eps_m, instance_min_pts, _updates,
    #   detection_cloud, _detection_frame_idx, _promoted_match_for_detection,
    #   _latest_detection_idx, clouds, _blacklist_instances,
    #   _vox_cache, _cache_cap,
    #   last_target_coord

    # ------------------------------------------------------------------
    # Overlap / seen detection
    # ------------------------------------------------------------------

    @staticmethod
    def _cloud_center_xy(cloud: np.ndarray) -> np.ndarray:
        if cloud.shape[0] == 0:
            return np.zeros(2, dtype=np.float32)
        inr = cloud[cloud[:, -1] == 1]
        pts = inr if inr.shape[0] > 0 else cloud
        return np.mean(pts[:, :2], axis=0).astype(np.float32)

    @staticmethod
    def _voxel_keys(pts_xyz: np.ndarray, voxel_m: float = 0.05) -> set:
        if pts_xyz.size == 0:
            return set()
        v = np.floor(pts_xyz / max(1e-6, float(voxel_m))).astype(np.int32)
        return set(map(tuple, v))

    def is_detection_seen(self, new_detection: np.ndarray, ref_instance: np.ndarray) -> bool:
        """Voxel-overlap test to decide whether two clouds represent the same instance.

        Quick pass if centres are very close; otherwise quantise both clouds
        into a voxel grid and compare intersection / min(|A|,|B|).
        """
        if new_detection.size == 0 or ref_instance.size == 0:
            return False

        # Fast pass when centres are very close
        c1 = self._cloud_center_xy(new_detection)
        c2 = self._cloud_center_xy(ref_instance)
        if float(np.linalg.norm(c1 - c2)) <= float(self.instance_eps_m) * 0.75:
            return True

        A = new_detection[:, :3].astype(np.float32)
        B = ref_instance[:, :3].astype(np.float32)

        Ncap = 5000
        if A.shape[0] > Ncap:
            A = A[np.random.choice(A.shape[0], Ncap, replace=False)]
        if B.shape[0] > Ncap:
            B = B[np.random.choice(B.shape[0], Ncap, replace=False)]

        from vlfm.config import load_config
        _inst_cfg = load_config().get("instance", {})
        vs = _inst_cfg.get("det_voxel_m", 0.05)
        dlt = _inst_cfg.get("det_overlap_min_ratio", 0.45)
        Sa = self._voxel_keys(A, vs)

        bid = id(ref_instance)
        Sb = self._vox_cache.get(bid)
        if Sb is None:
            Sb = self._voxel_keys(B, vs)
            if len(self._vox_cache) > int(getattr(self, "_cache_cap", 512)):
                self._vox_cache.clear()
            self._vox_cache[bid] = Sb

        if len(Sa) > 0 and len(Sb) > 0:
            inter = len(Sa.intersection(Sb))
            score = float(inter) / float(max(1, min(len(Sa), len(Sb))))
            if score >= dlt:
                return True

        return False

    # ------------------------------------------------------------------
    # Detection-buffer assignment
    # ------------------------------------------------------------------

    def _match_promoted_instance_idx(self, object_name: str, cloud: np.ndarray, topk: int = 5) -> int:
        insts = self.clouds.get(object_name, [])
        if len(insts) == 0:
            return -1
        c_new = self._cloud_center_xy(cloud)
        centers = np.vstack([self._cloud_center_xy(c) for c in insts])
        dists = np.linalg.norm(centers - c_new.reshape(1, 2), axis=1)
        order = list(np.argsort(dists))[: min(topk, len(insts))]
        for k in order:
            if self.is_detection_seen(cloud, insts[k]):
                return k
        return -1

    def _assign_detection_instance(
        self, object_name: str, cloud: np.ndarray, frame_id: Optional[int] = None
    ) -> Tuple[int, int, bool]:
        """Place *cloud* into the detection buffer with optional promoted-match tracking.

        Returns:
            (idx_in_detection_cloud, matched_promoted_idx, created_new_in_buffer)
        """
        matched_prom = self._match_promoted_instance_idx(object_name, cloud)
        lst = self.detection_cloud[object_name]
        frames = self._detection_frame_idx[object_name]
        pmidx = self._promoted_match_for_detection[object_name]

        # [A] Promoted match exists -- append and record linkage
        if matched_prom >= 0:
            lst.append(cloud)
            frames.append(int(frame_id) if frame_id is not None else -1)
            pmidx.append(int(matched_prom))
            return len(lst) - 1, int(matched_prom), False

        # [B] No promoted match -- try geometric merge inside the buffer
        if len(lst) > 0:
            c_new = self._cloud_center_xy(cloud)
            centers = np.vstack([self._cloud_center_xy(c) for c in lst])
            dists = np.linalg.norm(centers - c_new.reshape(1, 2), axis=1)
            order = list(np.argsort(dists))[: min(3, len(lst))]
            for k in order:
                if frame_id is not None and int(frames[k]) == int(frame_id):
                    continue  # same-frame guard
                if self.is_detection_seen(cloud, lst[k]):
                    lst[k] = np.concatenate([lst[k], cloud], axis=0)
                    try:
                        frames[k] = int(frame_id) if frame_id is not None else frames[k]
                    except Exception:
                        pass
                    if k < len(pmidx):
                        pmidx[k] = -1
                    return k, -1, False

        # [C] No merge -- append as new buffer entry
        lst.append(cloud)
        frames.append(int(frame_id) if frame_id is not None else -1)
        pmidx.append(-1)
        return len(lst) - 1, -1, True

    # ------------------------------------------------------------------
    # Commit / promote detection to clouds
    # ------------------------------------------------------------------

    def _adjust_latest_after_pop(self, object_name: str, popped_idx: int) -> None:
        """Maintain _latest_detection_idx after removing an entry from detection_cloud."""
        cur = self._latest_detection_idx.get(object_name, None)
        if cur is None:
            return
        cur = int(cur)
        if cur == popped_idx:
            self._latest_detection_idx.pop(object_name, None)
        elif cur > popped_idx:
            self._latest_detection_idx[object_name] = cur - 1

    def commit_detection_instance_to_clouds(
        self, object_name: str, instance_idx: int, set_last_coord: bool = True
    ) -> Tuple[bool, int]:
        """Promote a single detection instance to the confirmed cloud map."""
        lst = self.detection_cloud.get(object_name, [])
        if instance_idx is None or instance_idx < 0 or instance_idx >= len(lst):
            return (False, -1)

        pm = self._promoted_match_for_detection.get(object_name, [])
        match_idx = int(pm[instance_idx]) if (instance_idx < len(pm)) else -1

        inst = lst.pop(instance_idx)
        try:
            self._detection_frame_idx[object_name].pop(instance_idx)
        except Exception:
            pass
        try:
            self._promoted_match_for_detection[object_name].pop(instance_idx)
        except Exception:
            pass

        # Merge into matched promoted instance or append new
        if 0 <= match_idx < len(self.clouds.get(object_name, [])):
            self.clouds[object_name][match_idx] = np.concatenate(
                [self.clouds[object_name][match_idx], inst], axis=0
            )
            final_idx = match_idx
        else:
            self.clouds[object_name].append(inst)
            final_idx = int(len(self.clouds[object_name]) - 1)

        if set_last_coord:
            pts = inst[inst[:, -1] == 1]
            if pts.shape[0] == 0:
                pts = inst
            try:
                self.last_target_coord = np.mean(pts[:, :2], axis=0).astype(np.float32)
            except Exception:
                pass

        self._adjust_latest_after_pop(object_name, instance_idx)
        self._rebuild_instances_from_clouds(object_name)
        return (True, final_idx)

    def promoted_match_index(self, object_name: str, instance_idx: int) -> int:
        pm = self._promoted_match_for_detection.get(object_name, [])
        if instance_idx is None or instance_idx < 0 or instance_idx >= len(pm):
            return -1
        try:
            return int(pm[instance_idx])
        except Exception:
            return -1

    # ------------------------------------------------------------------
    # Instance bookkeeping (bootstrap / ensure / merge / rebuild / update)
    # ------------------------------------------------------------------

    def _ensure_instance_state(self) -> None:
        """Safely initialise instance-related members if missing."""
        if not hasattr(self, "_instances"):
            self._instances = defaultdict(list)
        if not hasattr(self, "instance_eps_m"):
            self.instance_eps_m = 0.5
        if not hasattr(self, "instance_min_pts"):
            self.instance_min_pts = 20
        if not hasattr(self, "_updates"):
            self._updates = 0

    def _rebuild_instances_from_clouds(self, object_name: str) -> None:
        """Make _instances[class] mirror self.clouds[class]."""
        self._ensure_instance_state()
        rebuilt: List[_Instance] = []
        for cl in self.clouds.get(object_name, []):
            if not isinstance(cl, np.ndarray) or cl.shape[0] == 0:
                continue
            c = self._cloud_center_xy(cl)
            rebuilt.append(
                _Instance(center_xy=c, n_points=int(cl.shape[0]), last_update=self._updates)
            )
        self._instances[object_name] = rebuilt

    # ------------------------------------------------------------------
    # Querying instances
    # ------------------------------------------------------------------

    def list_candidate_centers(
        self, object_name: str, eps: float = 0.35, min_points: int = 80
    ) -> List[np.ndarray]:
        """Return persisted instance centres (DBSCAN-free; args kept for compat)."""
        return [inst.center_xy.copy() for inst in self._instances.get(object_name, [])]

    # ------------------------------------------------------------------
    # Miscellaneous instance helpers
    # ------------------------------------------------------------------

    def _matches_any_blacklisted_instance(self, object_name: str, candidate: np.ndarray) -> bool:
        """Return True if *candidate* overlaps any blacklisted instance."""
        for b in self._blacklist_instances.get(object_name, []):
            if self.is_detection_seen(candidate, b):
                return True
        return False

    def detection_center_key(
        self, object_name: str, instance_idx: int
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        lst = self.detection_cloud.get(object_name, [])
        if instance_idx is None or instance_idx < 0 or instance_idx >= len(lst):
            return None, None
        center = self._cloud_center_xy(lst[instance_idx])
        return self._center_key(center), center

    def get_promoted_cloud(self, object_name: str, index: int) -> Union[None, np.ndarray]:
        insts = self.clouds.get(object_name, [])
        if not isinstance(insts, list) or index is None or index < 0 or index >= len(insts):
            return None
        return insts[index]

    def get_closest_point_on_promoted(
        self, object_name: str, prom_index: int, curr_position: np.ndarray
    ) -> Union[None, np.ndarray]:
        """Return the closest point (xy) on a promoted instance to the robot."""
        from vlfm.mapping._cloud_geometry import get_closest_point

        cl = self.get_promoted_cloud(object_name, int(prom_index))
        if cl is None or cl.shape[0] == 0:
            return None
        p = get_closest_point(cl, np.asarray(curr_position, np.float32), use_dbscan=self.use_dbscan)
        return p[:2].astype(np.float32)

    def promoted_center(self, object_name: str, index: int) -> Union[None, np.ndarray]:
        insts = self.clouds.get(object_name, [])
        if not isinstance(insts, list) or index < 0 or index >= len(insts):
            return None
        return self._cloud_center_xy(insts[index])
