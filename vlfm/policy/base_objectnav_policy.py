from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch import Tensor

from vlfm.config import load_config
from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.mapping.wall_map import WallMap
from vlfm.policy._detection import DetectionMixin
from vlfm.policy._instance_store import InstanceStoreMixin
from vlfm.policy._navigation import NavigationMixin
from vlfm.policy._object_map_update import ObjectMapUpdateMixin
from vlfm.policy._spatial_reasoning import SpatialReasoningMixin
from vlfm.policy._target_extraction import TargetExtractionMixin
from vlfm.policy._visualization import VisualizationMixin
from vlfm.policy._wall_and_room import WallAndRoomMixin
from vlfm.policy._ynu_defer import YNUDeferMixin
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.vlm.goal_clip import GoalClipClient
from vlfm.vlm.grounding_dino import GroundingDINOClient
from vlfm.vlm.llm_query import LLMQuery
from vlfm.vlm.sam import MobileSAMClient
from vlfm.vlm.vlm_query import VLMQuery
from vlfm.vlm.yolov7 import YOLOv7Client

try:
    from habitat_baselines.common.tensor_dict import TensorDict
    from vlfm.policy.base_policy import BasePolicy
except Exception:

    class BasePolicy:  # type: ignore
        pass


class BaseObjectNavPolicy(
    DetectionMixin,
    ObjectMapUpdateMixin,
    SpatialReasoningMixin,
    WallAndRoomMixin,
    YNUDeferMixin,
    InstanceStoreMixin,
    VisualizationMixin,
    NavigationMixin,
    TargetExtractionMixin,
    BasePolicy,
):
    _target_object: str = ""
    _policy_info: Dict[str, Any] = {}
    _object_masks: Union[np.ndarray, Any] = None
    _stop_action: Union[Tensor, Any] = None  # MUST BE SET BY SUBCLASS
    _observations_cache: Dict[str, Any] = {}
    _non_coco_caption = ""

    _nav_rgb_history: List[np.ndarray] = []
    _nav_clip_scores: List[float] = []
    _verify_with_vlm: bool = True
    _vlm_model: str = "qwen2.5vl:7b-fp16"
    _vlm_api_key: str = "ollama"
    _blacklist_radius_m: float = 0.5
    _target_description: str = ""

    _attr_questions: List[Dict[str, str]] = []
    _llm_model: str = "gpt-oss:20b"
    _use_bbox_for_vlm: bool = True
    _context_categories: List[str] = []
    _ig_high_categories: List[str] = []
    _context_relations: List[Dict[str, str]] = []
    _in_local_session: bool = False
    _local_anchor_xy: Union[np.ndarray, None] = None
    _local_deadline: int = 0
    _theta_intervals: List[Tuple[float, float]] = []
    _ctx_raw2canon: Dict[str, str] = {}
    _pending_attr_categories: Set[str] = set()
    _supported_rtypes = {"left", "right", "front", "behind", "near", "below"}

    def __init__(
        self,
        pointnav_policy_path: str,
        depth_image_shape: Tuple[int, int],
        pointnav_stop_radius: float,
        object_map_erosion_size: float,
        visualize: bool = True,
        compute_frontiers: bool = True,
        min_obstacle_height: float = 0.15,
        max_obstacle_height: float = 0.88,
        agent_radius: float = 0.18,
        obstacle_map_area_threshold: float = 1.5,
        hole_area_thresh: int = 100000,
        vqa_prompt: str = "Is this ",
        coco_threshold: float = 0.8,
        non_coco_threshold: float = 0.45,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Load config (YAML defaults + env var overrides)
        cfg = load_config()
        ports = cfg["ports"]
        vlm_cfg = cfg["vlm"]
        llm_cfg = cfg["llm"]
        ig_cfg = cfg["ig_weights"]
        nav_cfg = cfg["navigation"]
        ynu_cfg = cfg["ynu"]
        wall_cfg = cfg["wall_vis"]
        bench_cfg = cfg["benchmark_resolved"]

        # Service clients
        self._object_detector = GroundingDINOClient(port=ports["grounding_dino"])
        self._clip2itm = GoalClipClient(port=ports["clip2itm"])
        self._mobile_sam = MobileSAMClient(port=ports["sam"])
        self._coco_object_detector = YOLOv7Client(port=ports["yolov7"])

        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(erosion_size=object_map_erosion_size)
        self._depth_image_shape = tuple(depth_image_shape)
        self._pointnav_stop_radius = pointnav_stop_radius
        self._visualize = visualize

        self._num_steps = 0
        self._did_reset = False
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._called_stop = False
        self._compute_frontiers = compute_frontiers

        # VLM / LLM settings
        self._verify_with_vlm = vlm_cfg["verify_with_vlm"]
        self._vlm_model = vlm_cfg["model"]
        self._vlm_base_url = vlm_cfg["base_url"]
        self._vlm_api_key = vlm_cfg["api_key"]
        self._use_bbox_for_vlm = vlm_cfg["use_bbox"]
        self._llm_model = llm_cfg["model"]

        # Navigation
        self._blacklist_radius_m = nav_cfg["blacklist_radius_m"]
        self._relation_max_distance_m = nav_cfg["relation_max_distance_m"]

        if compute_frontiers:
            self._obstacle_map = ObstacleMap(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                area_thresh=obstacle_map_area_threshold,
                agent_radius=agent_radius,
                hole_area_thresh=hole_area_thresh,
            )
        self._context_categories: list = []
        self._vis_map: dict = {}
        self._sig_map: dict = {}
        self._ig_map: dict = {}

        # Information gain weights
        self._ig_w_vis = ig_cfg["vis"]
        self._ig_w_sig = ig_cfg["sig"]

        # Benchmark-resolved settings
        self.fallback_step = bench_cfg["fallback_step"]

        self._nav_ctx_scores: List[float] = []
        self._nav_rgb_history = []
        self._nav_clip_scores = []
        self._pending_attr_categories = set()

        self._wall_map = WallMap(size=1000, pixels_per_meter=20, hole_area_thresh=hole_area_thresh)

        # YNU defer settings
        self._ynu_unknown_pass_after = ynu_cfg["unknown_pass_after"]
        self._ynu_unknown_counts = {}
        self._ynu_passed_cats = set()
        self._ynu_defer_window = ynu_cfg["defer_window"]
        self._ynu_defer_active = False
        self._ynu_defer_start_step = 0
        self._ynu_buffer = []
        self._attr_defer_window = 5
        self._attr_defer_active = False
        self._attr_defer_start_step = 0
        self._attr_buffer = []

        self._inst_state_store: Dict = {}
        self._active_inst_key = None
        self._ynu_T_low: int = ynu_cfg["T_low"]
        self._ynu_T_high: int = ynu_cfg["T_high"]
        self._llm = LLMQuery(
            base_url=self._vlm_base_url,
            llm_model=self._llm_model,
            ig_w_vis=self._ig_w_vis,
            ig_w_sig=self._ig_w_sig,
        )

        self._vlm = VLMQuery(
            base_url=self._vlm_base_url,
            vlm_model=self._vlm_model,
            ynu_T_low=self._ynu_T_low,
            ynu_T_high=self._ynu_T_high,
        )
        self._inst_key_next: int = 0
        self._det2store_key: Dict = {}
        self._prom2store_key: Dict = {}

        self._frontier_lock_active = False
        self._frontier_lock_goal = None
        self._frontier_lock_reason = ""
        self._frontier_lock_store_key = None
        self.stage = None
        self._last_target_prom_idx: Union[int, None] = None
        self.fallback_store = None

        # Benchmark config flags
        self._cfg_benchmark = cfg["benchmark"]
        self._cfg_enable_vlm_exist_check = bench_cfg["enable_vlm_exist_check"]
        self._cfg_enable_depth_fov_correction = bench_cfg["enable_depth_fov_correction"]

        # Detection thresholds
        det_cfg = cfg["detection"]
        self._cfg_maxdepth_norm_th = det_cfg["maxdepth_norm_th"]
        self._cfg_maxdepth_skip_ratio = det_cfg["maxdepth_skip_ratio"]
        self._cfg_vlm_decision_threshold = det_cfg["vlm_decision_threshold"]

        # Spatial reasoning
        spatial_cfg = cfg["spatial"]
        self._cfg_rel_verify_max_inst_per_class = spatial_cfg["rel_verify_max_inst_per_class"]

        # Visualization
        vis_cfg = cfg["visualization"]
        self._cfg_depth_hfov_deg = vis_cfg["depth_hfov_deg"]
        self._cfg_rgb_hfov_deg = vis_cfg["rgb_hfov_deg"]

        # Wall visualization (PSL defaults)
        self._cfg_wall_vis_explored_mask = wall_cfg["explored_mask"]
        self._cfg_wall_vis_wall_color = wall_cfg["wall_color"]
        self._cfg_wall_vis_wall_thickness = wall_cfg["wall_thickness"]
        self._cfg_wall_vis_explored_color = wall_cfg["explored_color"]

    def _reset(self) -> None:
        self._target_object = ""
        self._target_object_attributes = {}
        self._pointnav_policy.reset()
        self._object_map.reset()
        self._last_goal = np.zeros(2)
        self._num_steps = 0
        self._done_initializing = False
        self._called_stop = False
        if self._compute_frontiers:
            self._obstacle_map.reset()
        self._did_reset = True
        self._nav_rgb_history = []
        self._nav_clip_scores = []
        self._attr_questions = []
        self._context_categories = []
        self._ig_high_categories = []
        self._context_relations = []
        self._in_local_session = False
        self._local_anchor_xy = None
        self._local_deadline = 0
        self._theta_intervals = []
        self._ctx_raw2canon = {}
        self._non_coco_caption = ""

        self._nav_ctx_scores: List[float] = []
        self._pending_attr_categories = set()
        self._wall_map.reset()
        self._ynu_unknown_counts = {}
        self._ynu_passed_cats = set()
        self._ynu_defer_active = False
        self._ynu_defer_start_step = 0
        self._ynu_buffer = []
        self._attr_defer_active = False
        self._attr_defer_start_step = 0
        self._attr_buffer = []

        self._inst_state_store = {}
        self._active_inst_key = None
        self._det2store_key = {}
        self._prom2store_key = {}
        self._inst_key_next = 0
        self._far_visit_goal = None

        self._frontier_lock_active = False
        self._frontier_lock_goal = None
        self._frontier_lock_reason = ""
        self._frontier_lock_store_key = None
        self._object_masks_vis = 0
        self.stage = None
        self._last_target_prom_idx = None
        self.fallback_store = None

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        """
        Starts the episode by 'initializing' and allowing robot to get its bearings.
        Then, explores the scene until it finds the target object.
        Once the target object is found, it navigates to the object.
        """
        self._pre_step(observations, masks)

        object_map_rgbd = self._observations_cache["object_map_rgbd"]
        detections = [
            self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy)
            for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
        ]
        for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd:
            self._update_wall_map_step(rgb, depth, tf, min_depth, max_depth, fx, fy)

        robot_xy = self._observations_cache["robot_xy"]
        goal = self._get_target_object_location(robot_xy)

        if goal is not None and self._frontier_lock_active:
            self._frontier_lock_active = False
            self._frontier_lock_goal = None
            self._frontier_lock_reason = ""
            self._frontier_lock_store_key = None

        progress_goal = goal
        if progress_goal is None and self._far_visit_goal is not None:
            progress_goal = self._far_visit_goal[:2]

        fb_goal = None
        if self.stage == "fallback":
            fb_goal = self.fallback_store
        if (goal is None) and (self._num_steps >= self.fallback_step) and (fb_goal is None):
            try:
                if self._object_map.has_scored_candidates(self._target_object):
                    scored_xy = self._object_map.best_scored_candidate(self._target_object)
                    if scored_xy is not None:
                        near_pt = self._closest_point_for_scored_center(np.asarray(scored_xy, np.float32).reshape(2), robot_xy)
                        if near_pt is not None:
                            fb_goal = near_pt
                        else:
                            fb_goal = np.asarray(scored_xy, np.float32).reshape(2)
                if fb_goal is None:
                    self.fallback_store = fb_goal
                    fb_goal = self._object_map.get_best_object(self._target_object, robot_xy)
            except Exception:
                fb_goal = None

        # --- Main mode selection ---
        if not self._done_initializing:
            mode = "initialize"
            pointnav_action = self._initialize()
        elif (goal is None) and self._frontier_lock_active and (self._frontier_lock_goal is not None):
            mode = "frontier_lock"
            pointnav_action = self._pointnav(self._frontier_lock_goal[:2], stop=True, stop_radius=1.2)
            if self._called_stop:
                self._called_stop = False
                self._frontier_lock_active = False
                self._frontier_lock_goal = None
                self._frontier_lock_reason = ""
                self._frontier_lock_store_key = None
                new_goal = self._get_target_object_location(robot_xy)
                if new_goal is None:
                    mode = "explore"
                    pointnav_action = self._explore(observations)
                else:
                    mode = "navigate"
                    pointnav_action = self._pointnav(new_goal[:2], stop=True)
        elif goal is None:
            if fb_goal is not None:
                mode = "fallback"
                pointnav_action = self._pointnav(fb_goal[:2], stop=True)
            elif self._far_visit_goal is not None:
                mode = "approach_far"
                pointnav_action = self._pointnav(self._far_visit_goal[:2], stop=True, stop_radius=1.2)
                if self._called_stop:
                    self._called_stop = False
                    self._far_visit_goal = None
                    new_goal = self._get_target_object_location(robot_xy)
                    if new_goal is None:
                        mode = "explore"
                        pointnav_action = self._explore(observations)
                    else:
                        mode = "navigate"
                        pointnav_action = self._pointnav(new_goal[:2], stop=True)
            else:
                mode = "explore"
                self._nav_rgb_history = []
                self._nav_clip_scores = []
                pointnav_action = self._explore(observations)
        else:
            mode = "navigate"
            pointnav_action = self._pointnav(goal[:2], stop=True)

        if mode == "navigate" and self._called_stop:
            try:
                rel_state = self._verify_relations_on_stop_joint(radius_m=3.0, require_joint_view=True, include_context_only=True)
            except Exception as e:
                print(f"[VERIFY] relation(joint) error: {e}")
                rel_state = "unknown"

            if (isinstance(rel_state, dict) and rel_state.get("state") == "fail"):
                self._called_stop = False
                try:
                    approx_xy = (self._object_map.last_target_coord
                                 if self._object_map.last_target_coord is not None
                                 else (goal[:2] if goal is not None else self._observations_cache["robot_xy"]))
                    self._object_map.blacklist_promoted_instance_near_xy(self._target_object, approx_xy)
                    stats = self._verify_relations_on_stop_joint(return_stats=True)
                    if isinstance(stats, dict):
                        self._object_map.record_candidate_ctxcount_score(self._target_object, approx_xy, float(stats.get("neighbor_count", 0)))
                        self._object_map.record_candidate_rel_score(self._target_object, approx_xy, float(stats.get("satisfied_count", 0)))
                except Exception:
                    pass
                self._nav_rgb_history = []
                self._nav_clip_scores = []
                self._nav_ctx_scores = []
                new_goal = self._get_target_object_location(robot_xy)
                if new_goal is None:
                    mode = "explore"
                    pointnav_action = self._explore(observations)
                else:
                    mode = "navigate"
                    pointnav_action = self._pointnav(new_goal[:2], stop=False)

        action_numpy = pointnav_action.detach().cpu().numpy()[0]
        if getattr(self, "_attr_defer_active", False):
            try:
                rgb_now = self._observations_cache["object_map_rgbd"][0][0]
                self._attr_collect_any(rgb_now, mask=None)
            except Exception:
                pass
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]
        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")

        if mode == "frontier_lock":
            self.stage = self._frontier_lock_reason
        else:
            self.stage = mode

        self._policy_info.update(self._get_policy_info(detections[0]))

        self._num_steps += 1

        self._observations_cache = {}
        self._did_reset = False

        return pointnav_action, rnn_hidden_states

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if not self._did_reset and masks[0] == 0:
            self._reset()
            self._extract_target_and_attributes(observations)

        try:
            self._cache_observations(observations)
        except IndexError as e:
            print(e)
            print("Reached edge of map, stopping.")
            raise StopIteration

        self._policy_info = {}

    def _initialize(self) -> Tensor:
        raise NotImplementedError

    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError

    def _cache_observations(self, observations) -> None:
        """Extracts the rgb, depth, and camera transform from the observations."""
        raise NotImplementedError

    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """Infers the depth image from the rgb image."""
        raise NotImplementedError


@dataclass
class VLFMConfig:
    name: str = "HabitatITMPolicy"
    text_prompt: str = "Seems like there is a target_object ahead."
    pointnav_policy_path: str = "data/pointnav_weights.pth"
    depth_image_shape: Tuple[int, int] = (224, 224)
    pointnav_stop_radius: float = 0.75
    use_max_confidence: bool = False
    object_map_erosion_size: int = 5
    exploration_thresh: float = 0.0
    obstacle_map_area_threshold: float = 0.5
    min_obstacle_height: float = 0.61
    max_obstacle_height: float = 0.88
    hole_area_thresh: int = 100000
    use_vqa: bool = False
    vqa_prompt: str = "Is this "
    coco_threshold: float = 0.8
    non_coco_threshold: float = 0.45
    agent_radius: float = 0.18
    verify_with_vlm: bool = True
    blacklist_radius_m: float = 1.0
    vlm_model: str = "qwen2.5vl:7b-fp16"
    vlm_api_key: str = "ollama"
    # --- Unified config fields for COIN/PSL divergences ---
    fallback_step: int = 400
    enable_vlm_exist_check: bool = True
    enable_depth_fov_correction: bool = True
    enable_depth_reprojection: bool = True
    wall_vis_explored_mask: bool = False
    wall_vis_wall_color: Optional[Tuple[int, int, int]] = None
    wall_vis_wall_thickness: Optional[int] = None
    wall_vis_explored_color: Optional[Tuple[int, int, int]] = None

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        return [f.name for f in fields(VLFMConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="vlfm_config_base", node=VLFMConfig())
