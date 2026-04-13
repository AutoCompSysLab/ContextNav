import hashlib
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from frontier_exploration.utils.general_utils import xyz_to_habitat
from habitat.utils.common import flatten_dict
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.maps import MAP_TARGET_POINT_INDICATOR
from habitat.utils.visualizations.utils import overlay_text_to_image
from habitat_baselines.common.tensor_dict import TensorDict

from vlfm.utils.geometry_utils import transform_points
from vlfm.utils.img_utils import (
    reorient_rescale_map,
    resize_image,
    resize_images,
    rotate_image,
)
from vlfm.utils.visualization import add_text_to_image, pad_images

class HabitatVis:
    def __init__(self) -> None:
        self.rgb: List[np.ndarray] = []
        self.depth: List[np.ndarray] = []
        self.maps: List[np.ndarray] = []
        self.vis_maps: List[List[np.ndarray]] = []
        self.texts: List[List[str]] = []
        self.using_vis_maps = False
        self.using_annotated_rgb = False
        self.using_annotated_depth = False

    def reset(self) -> None:
        self.rgb = []
        self.depth = []
        self.maps = []
        self.vis_maps = []
        self.texts = []
        self.using_annotated_rgb = False
        self.using_annotated_depth = False

    def collect_data(
        self,
        observations: TensorDict,
        infos: List[Dict[str, Any]],
        policy_info: List[Dict[str, Any]],
    ) -> None:
        assert len(infos) == 1, "Only support one environment for now"

        if "annotated_depth" in policy_info[0]:
            depth = policy_info[0]["annotated_depth"]
            self.using_annotated_depth = True
        else:
            depth = (observations["depth"][0].cpu().numpy() * 255.0).astype(np.uint8)
            depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        self.depth.append(depth)

        if "annotated_rgb" in policy_info[0]:
            rgb = policy_info[0]["annotated_rgb"]
            self.using_annotated_rgb = True
        else:
            rgb = observations["rgb"][0].cpu().numpy()
        self.rgb.append(rgb)

        # Visualize target point cloud on the map
        color_point_cloud_on_map(infos, policy_info)

        map = maps.colorize_draw_agent_and_fit_to_height(infos[0]["top_down_map"], self.depth[0].shape[0])
        map = self._center_on_nonwhite(map)
        self.maps.append(map)

        vis_map_imgs = []
        for vkey in ["value_map"]:
            if vkey in policy_info[0]:
                img = self._reorient_rescale_habitat_map(infos, policy_info[0][vkey])
                if vkey == "wall_map":
                    legend = policy_info[0].get("wall_legend_items", [])
                    start_yaw = float(infos[0].get("start_yaw", 0.0))
                    img = self._overlay_wall_labels(img, legend, start_yaw)
                vis_map_imgs.append(img)
        if vis_map_imgs:
            self.using_vis_maps = True
            self.vis_maps.append(vis_map_imgs)
        text = [
            policy_info[0][text_key]
            for text_key in policy_info[0].get("render_below_images", [])
            if text_key in policy_info[0]
        ]
        if policy_info[0].get("context_categories", "") != []:
            text.append("ctx: " + " ".join(policy_info[0].get("context_categories", "")))

        if policy_info[0].get("ig_high_categories", "") != []:
            text.append("ig: " + " ".join(policy_info[0].get("ig_high_categories", "")))
        text.append(policy_info[0].get("stage_text", ""))
        self.texts.append(text)

    def flush_frames(self, failure_cause: str) -> List[np.ndarray]:
        """Flush all frames and return them"""
        # Because the annotated frames are actually one step delayed, pop the first one
        # and add a placeholder frame to the end (gets removed anyway)
        if self.using_annotated_rgb:
            self.rgb.append(self.rgb.pop(0))
        if self.using_annotated_depth:
            self.depth.append(self.depth.pop(0))
        if self.using_vis_maps:  # Cost maps are also one step delayed
            self.vis_maps.append(self.vis_maps.pop(0))

        frames = []
        num_frames = len(self.depth) - 1  # last frame is from next episode, remove it
        for i in range(num_frames):
            frame = self._create_frame(
                self.depth[i],
                self.rgb[i],
                self.maps[i],
                self.vis_maps[i],
            )
            frames.append(frame)

        if len(frames) > 0:
            frames = pad_images(frames, pad_from_top=True)

        frames = [resize_image(f, 480 * 2) for f in frames]

        if len(frames) > 0:
            # Crop all frames to the non-black bounding box of the first frame
            r0, c0, r1, c1 = self._bbox_of_nonblack(frames[0])
            frames = [f[r0 : r1 + 1, c0 : c1 + 1] for f in frames]

        self.reset()

        return frames

    @staticmethod
    def _reorient_rescale_habitat_map(
        infos: List[Dict[str, Any]],
        vis_map: np.ndarray,
    ) -> np.ndarray:
        # Rotate the cost map to match the agent's orientation at the start
        start_yaw = infos[0]["start_yaw"]
        if start_yaw != 0.0:
            vis_map = rotate_image(vis_map, start_yaw, border_value=(255, 255, 255))

        # Rotate the image 90 degrees if the corresponding map is taller than it is wide
        habitat_map = infos[0]["top_down_map"]["map"]
        if habitat_map.shape[0] > habitat_map.shape[1]:
            vis_map = np.rot90(vis_map, 1)

        vis_map = reorient_rescale_map(vis_map)

        # Center the non-white content region
        vis_map = HabitatVis._center_on_nonwhite(vis_map)

        return vis_map

    @staticmethod
    def _create_frame(
        depth: np.ndarray,
        rgb: np.ndarray,
        map: np.ndarray,
        vis_map_imgs: List[np.ndarray],
    ) -> np.ndarray:
        # Use the first vis_map (value_map) if available, otherwise fall back to topdown map
        if len(vis_map_imgs) > 0:
            value_map = vis_map_imgs[0]
        else:
            value_map = map

        # Match heights then concatenate horizontally
        rgb_resized, value_resized = resize_images(
            [rgb, value_map],
            match_dimension="height",
        )
        frame = np.hstack((rgb_resized, value_resized))

        return frame

    @staticmethod
    def _stable_rgb_for_category(name: str) -> tuple[int, int, int]:
        """Hash-based deterministic color (50-230 range), returned in RGB order."""
        h = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16) & 0xFFFFFFFF
        rng = np.random.RandomState(h)
        b, g, r = [int(x) for x in rng.randint(50, 230, size=3)]
        return (r, g, b)

    def _overlay_wall_labels(self, vis_map: np.ndarray, legend_items: list[str], start_yaw: float) -> np.ndarray:
        """Overlay category legend onto the wall map, anchored near the start-yaw quadrant."""
        if not isinstance(legend_items, list) or len(legend_items) == 0:
            return vis_map
        H, W = vis_map.shape[:2]
        r0, c0, r1, c1 = self._bbox_of_nonwhite(vis_map)
        pad = 10
        # Normalize yaw to [0, 2pi)
        ang = (float(start_yaw) + 2.0 * np.pi) % (2.0 * np.pi)
        # Choose anchor corner based on yaw quadrant
        if ang < np.pi/4 or ang >= 7*np.pi/4:         # East
            x0, y0 = c0 + pad, r0 + pad
        elif ang < 3*np.pi/4:                         # North
            x0, y0 = c0 + pad, max(r1 - pad - 14, r0 + pad)
        elif ang < 5*np.pi/4:                         # West
            x0, y0 = max(c1 - pad - 14, c0 + pad), max(r1 - pad - 14, r0 + pad)
        else:                                         # South
            x0, y0 = max(c1 - pad - 14, c0 + pad), r0 + pad
        dy = 18
        for i, cat in enumerate(legend_items):
            color = self._stable_rgb_for_category(cat)
            y = min(max(y0 + i*dy, 0), H-14)
            x = min(max(x0, 0), W-14)
            cv2.rectangle(vis_map, (x, y), (x + 14, y + 14), color, -1)
            cv2.putText(
                vis_map, cat[:18],
                (x + 20, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA
            )
        return vis_map

    @staticmethod
    def _bbox_of_nonwhite(img: np.ndarray) -> tuple[int, int, int, int]:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        nz = np.argwhere(gray != 255)
        if nz.size == 0:
            return 0, 0, img.shape[0] - 1, img.shape[1] - 1
        min_r, min_c = np.min(nz, axis=0)
        max_r, max_c = np.max(nz, axis=0)
        return int(min_r), int(min_c), int(max_r), int(max_c)

    @staticmethod
    def _center_on_nonwhite(
        img: np.ndarray,
        pad: int = 10,
    ) -> np.ndarray:
        """Crop to non-white bounding box and center on a square canvas."""
        H, W = img.shape[:2]
        r0, c0, r1, c1 = HabitatVis._bbox_of_nonwhite(img)

        # Add padding around bbox
        r0 = max(r0 - pad, 0)
        c0 = max(c0 - pad, 0)
        r1 = min(r1 + pad, H - 1)
        c1 = min(c1 + pad, W - 1)

        cropped = img[r0 : r1 + 1, c0 : c1 + 1]
        h, w = cropped.shape[:2]

        # Create square canvas and center the cropped content
        side = max(h, w)
        canvas = np.full((side, side, 3), 255, dtype=img.dtype)
        top = (side - h) // 2
        left = (side - w) // 2
        canvas[top : top + h, left : left + w] = cropped
        return canvas

    @staticmethod
    def _bbox_of_nonblack(
        img: np.ndarray,
        thresh: int = 1,
    ) -> tuple[int, int, int, int]:
        """Return the bounding box of non-black pixels (grayscale > thresh)."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        nz = np.argwhere(gray > thresh)
        if nz.size == 0:
            h, w = img.shape[:2]
            return 0, 0, h - 1, w - 1

        min_r, min_c = np.min(nz, axis=0)
        max_r, max_c = np.max(nz, axis=0)
        return int(min_r), int(min_c), int(max_r), int(max_c)


def sim_xy_to_grid_xy(
    upper_bound: Tuple[int, int],
    lower_bound: Tuple[int, int],
    grid_resolution: Tuple[int, int],
    sim_xy: np.ndarray,
    remove_duplicates: bool = True,
) -> np.ndarray:
    """Converts simulation coordinates to grid coordinates.

    Args:
        upper_bound (Tuple[int, int]): The upper bound of the grid.
        lower_bound (Tuple[int, int]): The lower bound of the grid.
        grid_resolution (Tuple[int, int]): The resolution of the grid.
        sim_xy (np.ndarray): A numpy array of 2D simulation coordinates.
        remove_duplicates (bool): Whether to remove duplicate grid coordinates.

    Returns:
        np.ndarray: A numpy array of 2D grid coordinates.
    """
    grid_size = np.array(
        [
            abs(upper_bound[1] - lower_bound[1]) / grid_resolution[0],
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
        ]
    )
    grid_xy = ((sim_xy - lower_bound[::-1]) / grid_size).astype(int)

    if remove_duplicates:
        grid_xy = np.unique(grid_xy, axis=0)

    return grid_xy


def color_point_cloud_on_map(infos: List[Dict[str, Any]], policy_info: List[Dict[str, Any]]) -> None:
    if len(policy_info[0]["target_point_cloud"]) == 0:
        return

    upper_bound = infos[0]["top_down_map"]["upper_bound"]
    lower_bound = infos[0]["top_down_map"]["lower_bound"]
    grid_resolution = infos[0]["top_down_map"]["grid_resolution"]
    tf_episodic_to_global = infos[0]["top_down_map"]["tf_episodic_to_global"]

    cloud_episodic_frame = policy_info[0]["target_point_cloud"][:, :3]
    cloud_global_frame_xyz = transform_points(tf_episodic_to_global, cloud_episodic_frame)
    cloud_global_frame_habitat = xyz_to_habitat(cloud_global_frame_xyz)
    cloud_global_frame_habitat_xy = cloud_global_frame_habitat[:, [2, 0]]

    grid_xy = sim_xy_to_grid_xy(
        upper_bound,
        lower_bound,
        grid_resolution,
        cloud_global_frame_habitat_xy,
        remove_duplicates=True,
    )

    new_map = infos[0]["top_down_map"]["map"].copy()
    new_map[grid_xy[:, 0], grid_xy[:, 1]] = MAP_TARGET_POINT_INDICATOR

    infos[0]["top_down_map"]["map"] = new_map


