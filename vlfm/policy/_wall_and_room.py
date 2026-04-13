# Wall map and room logic mixin for BaseObjectNavPolicy

from typing import Union

import cv2
import numpy as np


class WallAndRoomMixin:
    """Wall map update, wall-line blocking, same-room check, and frontier room filtering."""

    def _update_wall_map_step(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> None:
        self._wall_map.update_from_depth(
            depth=depth,
            tf_camera_to_episodic=tf_camera_to_episodic,
            min_depth=min_depth,
            max_depth=max_depth,
            fx=fx,
            fy=fy,
        )

    def _wall_line_blocked(self, a_xy: np.ndarray, b_xy: np.ndarray, clearance_m: float = 0.05) -> bool:
        """
        Check if the line a->b crosses a wall in the wall occupancy map.
        Returns True if blocked (different room), False otherwise.
        """
        try:
            occ = self._wall_map.get_occupancy().astype(np.uint8)
            pts = np.vstack([
                np.asarray(a_xy, np.float32).reshape(1, 2),
                np.asarray(b_xy, np.float32).reshape(1, 2),
            ])
            px = self._wall_map._xy_to_px(pts)
            x1, y1 = int(px[0, 0]), int(px[0, 1])
            x2, y2 = int(px[1, 0]), int(px[1, 1])
            h, w = occ.shape[:2]
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            ray = np.zeros_like(occ, dtype=np.uint8)
            th = max(1, int(round(clearance_m * float(self._wall_map.pixels_per_meter))) * 2 + 1)
            cv2.line(ray, (x1, y1), (x2, y2), color=255, thickness=th)
            return bool(np.any((ray > 0) & (occ > 0)))
        except Exception:
            return False

    def _same_room_by_wall(self, a_xy: np.ndarray, b_xy: np.ndarray) -> bool:
        """a and b are in the same room if no wall blocks the line between them."""
        return not self._wall_line_blocked(a_xy, b_xy)

    def _choose_same_room_frontier(self, anchor_xy: np.ndarray) -> Union[None, np.ndarray]:
        """Find the closest frontier in the same room as both the anchor and the robot."""
        fts = self._observations_cache.get("frontier_sensor", np.array([]))
        if not isinstance(fts, np.ndarray) or fts.size == 0:
            return None
        robot_xy = self._observations_cache["robot_xy"]
        keep = []
        for ft in fts:
            if not self._wall_line_blocked(anchor_xy, ft) and not self._wall_line_blocked(robot_xy, ft):
                keep.append(np.asarray(ft, np.float32))
        if len(keep) == 0:
            return None
        d = np.linalg.norm(np.vstack(keep) - np.asarray(anchor_xy, np.float32).reshape(1, 2), axis=1)
        return keep[int(np.argmin(d))]
