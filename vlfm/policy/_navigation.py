# Navigation mixin for BaseObjectNavPolicy

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from vlfm.obs_transformers.utils import image_resize
from vlfm.utils.geometry_utils import rho_theta


class NavigationMixin:
    """Point navigation, approach, frontier lock, and target location logic."""

    def _pointnav(self, goal: np.ndarray, stop: bool = False, stop_radius: Optional[float] = None) -> Tensor:
        """
        Navigate toward goal using the pre-trained PointNav policy.
        """
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                masks = torch.zeros_like(masks)
            self._last_goal = goal
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info["rho_theta"] = np.array([rho, theta])
        radius = float(stop_radius) if (stop and stop_radius is not None) else float(self._pointnav_stop_radius)
        if rho < radius and stop:
            self._called_stop = True
            return self._stop_action
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        return action

    def _lock_frontier(self, goal_xy: np.ndarray, reason: str = "", store_key: Optional[str] = None) -> None:
        """Force navigation to a frontier. Automatically released when a verified target appears."""
        self._frontier_lock_active = True
        self._frontier_lock_goal = np.asarray(goal_xy, np.float32).reshape(2)
        self._frontier_lock_reason = str(reason or "")
        self._frontier_lock_store_key = store_key
        self._far_visit_goal = None

    def _closest_point_for_scored_center(self, center_xy: np.ndarray, robot_xy: np.ndarray) -> Optional[np.ndarray]:
        """
        Given a scored candidate center, find the closest cloud point to the robot
        across promoted, detection, and blacklisted instances.
        """
        cx = np.asarray(center_xy, np.float32).reshape(2)
        rxy = np.asarray(robot_xy, np.float32).reshape(2)
        try:
            k_scored = self._object_map._center_key(cx)
        except Exception:
            k_scored = f"{round(float(cx[0]), 2):.2f},{round(float(cx[1]), 2):.2f}"
        tol = float(getattr(self._object_map, "instance_eps_m", 0.35)) * 1.25

        def _match_center(cand_cloud: np.ndarray) -> bool:
            if not isinstance(cand_cloud, np.ndarray) or cand_cloud.shape[0] == 0:
                return False
            c = self._object_map._cloud_center_xy(cand_cloud)
            try:
                if self._object_map._center_key(c) == k_scored:
                    return True
            except Exception:
                pass
            return float(np.linalg.norm(np.asarray(c, np.float32) - cx)) <= tol

        # (1) Promoted instances (clouds)
        insts = self._object_map.clouds.get(self._target_object, [])
        best_pt, best_d = None, float("inf")
        for i, cl in enumerate(insts):
            if not _match_center(cl):
                continue
            pt = self._object_map.get_closest_point_on_promoted(self._target_object, int(i), rxy)
            if pt is None:
                continue
            d = float(np.linalg.norm(np.asarray(pt, np.float32) - rxy))
            if d < best_d:
                best_d, best_pt = d, np.asarray(pt, np.float32).reshape(2)
        if best_pt is not None:
            return best_pt

        # (2) Detection cloud
        det_list = self._object_map.detection_cloud.get(self._target_object, [])
        best_pt, best_d = None, float("inf")
        for cl in det_list:
            if not _match_center(cl):
                continue
            p3 = self._object_map.get_closest_point(cl, rxy)
            pt = np.asarray(p3[:2], np.float32)
            d = float(np.linalg.norm(pt - rxy))
            if d < best_d:
                best_d, best_pt = d, pt
        if best_pt is not None:
            return best_pt

        # (3) Blacklisted instances
        bl = self._object_map._blacklist_instances.get(self._target_object, [])
        best_pt, best_d = None, float("inf")
        for cl in bl:
            if not _match_center(cl):
                continue
            p3 = self._object_map.get_closest_point(cl, rxy)
            pt = np.asarray(p3[:2], np.float32)
            d = float(np.linalg.norm(pt - rxy))
            if d < best_d:
                best_d, best_pt = d, pt
        if best_pt is not None:
            return best_pt

        return None

    def _get_target_object_location(self, position: np.ndarray):
        """Get navigation goal for the target object, preferring context-supported instances."""
        if not self._object_map.has_object(self._target_object):
            return None

        # Try relation-context distance sum best instance
        prom_idx = None
        try:
            prom_idx = self._object_map.best_promoted_index_by_relctx_sum(self._target_object)
        except Exception:
            prom_idx = None
        if prom_idx is not None:
            closest_xy = self._object_map.get_closest_point_on_promoted(
                self._target_object, int(prom_idx), position
            )
            if closest_xy is None:
                inst_cloud = self._object_map.get_promoted_cloud(self._target_object, int(prom_idx))
                if inst_cloud is not None and inst_cloud.shape[0] > 0:
                    c = np.mean(inst_cloud[:, :2], axis=0).astype(np.float32)
                    closest_xy = c
            return closest_xy
        else:
            closest_xy = None

        if closest_xy is not None:
            chosen = np.asarray(closest_xy, np.float32).reshape(2)
            lt = self._object_map.last_target_coord
            if lt is None:
                self._object_map.last_target_coord = chosen.copy()
                self._last_target_prom_idx = int(prom_idx) if prom_idx is not None else None
                return chosen
            last_idx = self._last_target_prom_idx
            if last_idx is None and lt is not None:
                try:
                    last_idx = self._nearest_prom_index_to_xy(lt)
                except Exception:
                    last_idx = None
            same_instance = (prom_idx is not None) and (last_idx is not None) and (int(prom_idx) == int(last_idx))
            if same_instance:
                delta = float(np.linalg.norm(chosen - lt))
                far_from_robot = float(np.linalg.norm(np.asarray(position[:2], np.float32) - chosen)) > 2.0
                if delta < 0.1:
                    return lt
                elif delta < 0.5 and far_from_robot:
                    return lt
            self._object_map.last_target_coord = chosen.copy()
            self._last_target_prom_idx = int(prom_idx) if prom_idx is not None else None
            return chosen

        # Fallback: closest promoted instance
        return self._object_map.get_best_object(self._target_object, position)

    def _nearest_prom_index_to_xy(self, xy: np.ndarray) -> Union[int, None]:
        """Find the promoted instance index closest to the given xy."""
        insts = self._object_map.clouds.get(self._target_object, [])
        if not isinstance(insts, list) or len(insts) == 0:
            return None
        xy = np.asarray(xy, np.float32).reshape(2)
        best_i, best_d = None, float("inf")
        for i, inst in enumerate(insts):
            if not isinstance(inst, np.ndarray) or inst.shape[0] == 0:
                continue
            c = np.mean(inst[:, :2], axis=0).astype(np.float32)
            d = float(np.linalg.norm(c - xy))
            if d < best_d:
                best_d, best_i = d, i
        return best_i
