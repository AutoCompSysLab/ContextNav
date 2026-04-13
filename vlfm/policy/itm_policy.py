import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.detections import ObjectDetections

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._itm = self._clip2itm
        self._text_prompt = text_prompt
        self._value_map: ValueMap = ValueMap(
            value_channels=1,
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        self._acyclic_enforcer = AcyclicEnforcer()
        self.fallback_store = None
        self.stage = None

    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)
        self.fallback_store = None
        self.stage = None

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            robot_xy = self._observations_cache["robot_xy"]
            print("[EXPLORE] No frontiers; trying fallback candidates.")
            # Try best scored candidate first
            if self._object_map.has_scored_candidates(self._target_object):
                scored_xy = self._object_map.best_scored_candidate(self._target_object)
                if scored_xy is not None:
                    near_pt = self._closest_point_for_scored_center(
                        np.asarray(scored_xy, np.float32).reshape(2), robot_xy
                    )
                    fb_goal = near_pt if near_pt is not None else np.asarray(scored_xy, np.float32).reshape(2)
                    self.fallback_store = fb_goal
                    self.stage = "fallback"
                    return self._pointnav(fb_goal[:2], stop=True)
            # Try accumulated instance centers
            center = self._object_map.get_best_object(self._target_object, robot_xy)
            if center is not None:
                os.environ["DEBUG_INFO"] = "Fallback->best_known_center"
                return self._pointnav(center[:2], stop=True)
            print("[EXPLORE] No fallback candidates; stopping.")
            return self._stop_action

        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, stop=False)
        return pointnav_action

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map."""
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        os.environ["DEBUG_INFO"] = ""
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None
            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    curr_index = idx
                    break
            if curr_index is None:
                closest_index = closest_point_within_threshold(
                    sorted_pts, self._last_frontier, threshold=0.5
                )
                if closest_index != -1:
                    curr_index = closest_index
            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    print("Sticking to last point.")
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index

        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

        return best_frontier, best_value

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if not self._visualize:
            return policy_info

        markers = []
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info

    def _update_value_map(self) -> None:
        goal_attrs = self._target_object_attributes
        # PSL provides a dict with intrinsic/extrinsic lists; COIN provides a string
        if isinstance(goal_attrs, dict):
            t_full = " ".join(
                goal_attrs.get("intrinsic_attributes", [])
                + goal_attrs.get("extrinsic_attributes", [])
            )
        else:
            t_full = str(goal_attrs)

        for (rgb, depth, tf, min_depth, max_depth, fov) in self._observations_cache["value_map_rgbd"]:
            c = float(self._itm.cosine(rgb, t_full))
            self._value_map.update_map(np.array([c], dtype=np.float32), depth, tf, min_depth, max_depth, fov)
        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError


class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        if self._visualize:
            self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _reset(self) -> None:
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        rgb = self._observations_cache["object_map_rgbd"][0][0]
        text = self._text_prompt.replace("target_object", self._target_object)
        self._frontier_map.update(frontiers, rgb, text)  # type: ignore
        return self._frontier_map.sort_waypoints()


class ITMPolicyV2(BaseITMPolicy):
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)
        return sorted_frontiers, sorted_values


class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            first_channel = arr[:, :, 0]
            max_values = np.max(arr, axis=2)
            mask = first_channel > exploration_thresh
            result = np.where(mask, first_channel, max_values)
            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(
            frontiers, 0.5, reduce_fn=self._reduce_values
        )
        return sorted_frontiers, sorted_values

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """Reduce per-frontier (target_value, exploration_value) tuples to single values.

        If the max target_value is below the exploration threshold, return
        exploration values; otherwise return target values.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)
        if max_target_value < self._exploration_thresh:
            return [v[1] for v in values]
        else:
            return [v[0] for v in values]
