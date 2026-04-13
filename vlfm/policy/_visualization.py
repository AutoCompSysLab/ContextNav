# Visualization mixin for BaseObjectNavPolicy

import math
import os
from typing import Any, Dict

import cv2
import numpy as np

from vlfm.vlm.grounding_dino import ObjectDetections


class VisualizationMixin:
    """Policy info assembly, depth/wall visualization, and navigation frame recording."""

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        if self._object_map.has_object(self._target_object):
            target_point_cloud = self._object_map.get_target_cloud(self._target_object)
        else:
            target_point_cloud = np.array([])
        policy_info = {
            "target_object": self._target_object.split("|")[0],
            "gps": str(self._observations_cache["robot_xy"] * np.array([1, -1])),
            "yaw": np.rad2deg(self._observations_cache["robot_heading"]),
            "target_detected": self._object_map.has_object(self._target_object),
            "target_point_cloud": target_point_cloud,
            "nav_goal": self._last_goal,
            "stop_called": self._called_stop,
            "render_below_images": [
                "target_object",
            ],
            "stage_text": (self.stage if self.stage else ""),
            "context_categories": list(self._context_categories or []),
            "ig_high_categories": list(self._ig_high_categories or []),
        }

        if not self._visualize:
            return policy_info

        enable_fov_correction = getattr(self, '_cfg_enable_depth_fov_correction', True)

        if enable_fov_correction:
            depth_norm = self._observations_cache["object_map_rgbd"][0][1]  # 0..1
            H, W = depth_norm.shape[:2]
            depth_hfov_deg = self._cfg_depth_hfov_deg
            rgb_hfov_deg = self._cfg_rgb_hfov_deg
            s = math.tan(math.radians(rgb_hfov_deg) / 2.0) / max(1e-6, math.tan(math.radians(depth_hfov_deg) / 2.0))
            new_w = max(1, int(round(W * s)))
            new_h = max(1, int(round(H * s)))
            x0 = max(0, (W - new_w) // 2)
            y0 = max(0, (H - new_h) // 2)
            depth_cropped = depth_norm[y0:y0 + new_h, x0:x0 + new_w]
            depth_resized = cv2.resize(depth_cropped, (W, H), interpolation=cv2.INTER_NEAREST)
            annotated_depth = (depth_resized * 255).astype(np.uint8)
        else:
            annotated_depth = (self._observations_cache["object_map_rgbd"][0][1] * 255).astype(np.uint8)

        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if self._object_masks_vis.sum() > 0:
            contours, _ = cv2.findContours(self._object_masks_vis, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(detections.annotated_frame, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        else:
            annotated_rgb = self._observations_cache["object_map_rgbd"][0][0]
        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        if self._compute_frontiers:
            policy_info["obstacle_map"] = cv2.cvtColor(self._obstacle_map.visualize(), cv2.COLOR_BGR2RGB)

            # Build wall_map visualization kwargs from config
            wall_vis_kwargs = dict(
                object_clouds=self._object_map.clouds,
                draw_centers=True,
                draw_legend=False,
            )
            # Optional PSL-style params from config
            cfg_explored_mask = getattr(self, '_cfg_wall_vis_explored_mask', False)
            cfg_wall_color = getattr(self, '_cfg_wall_vis_wall_color', None)
            cfg_wall_thickness = getattr(self, '_cfg_wall_vis_wall_thickness', None)
            cfg_explored_color = getattr(self, '_cfg_wall_vis_explored_color', None)

            if cfg_explored_mask:
                wall_vis_kwargs["explored_mask"] = self._obstacle_map.explored_area
            if cfg_wall_color is not None:
                wall_vis_kwargs["wall_color"] = tuple(cfg_wall_color)
            if cfg_wall_thickness is not None:
                wall_vis_kwargs["wall_thickness_px"] = int(cfg_wall_thickness)
            if cfg_explored_color is not None:
                wall_vis_kwargs["explored_color"] = tuple(cfg_explored_color)

            policy_info["wall_map"] = cv2.cvtColor(
                self._wall_map.visualize_with_objects(**wall_vis_kwargs),
                cv2.COLOR_BGR2RGB,
            )
            policy_info["wall_legend_items"] = sorted(
                [k for k, v in self._object_map.clouds.items() if isinstance(v, list) and len(v) > 0]
            )

        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        image_np = policy_info["annotated_rgb"].astype('uint8')
        #plt.imsave('current_view.png', image_np)
        image_np = policy_info["annotated_depth"].astype('uint8')
        #plt.imsave('current_depth.png', image_np)
        image_wall = policy_info["wall_map"].astype('uint8')
        #plt.imsave('current_wall.png', image_wall)

        return policy_info

