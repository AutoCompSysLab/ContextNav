"""Blacklisting and multi-signal candidate scoring for ObjectPointCloudMap."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np


class BlacklistAndScoringMixin:
    """Mixed into ObjectPointCloudMap for blacklist and scoring logic."""

    # ------------------------------------------------------------------
    # Center-key utility
    # ------------------------------------------------------------------

    @staticmethod
    def _center_key(center_xy: np.ndarray) -> str:
        c = np.asarray(center_xy, dtype=np.float32).reshape(2)
        c = np.round(c, 2)
        return f"{c[0]:.2f},{c[1]:.2f}"

    # ------------------------------------------------------------------
    # Blacklisting
    # ------------------------------------------------------------------

    def blacklist_region(self, object_name: str, center_xy: np.ndarray, radius_m: float) -> None:
        """Register a failed-search coordinate and prune existing clouds."""
        center_xy = np.asarray(center_xy).astype(np.float32).reshape(2)
        self._blacklist[object_name].append((center_xy, float(radius_m)))

        if object_name in self.clouds:
            new_list: List[np.ndarray] = []
            for inst in self.clouds[object_name]:
                if inst.shape[0] == 0:
                    continue
                xy = inst[:, :2]
                mask_valid = np.ones((xy.shape[0],), dtype=bool)
                for cxy, r in self._blacklist[object_name]:
                    dist = np.linalg.norm(xy - cxy[None, :], axis=1)
                    mask_valid &= (dist >= r)
                inst2 = inst[mask_valid]
                if inst2.shape[0] > 0:
                    new_list.append(inst2)
            self.clouds[object_name] = new_list
            self._rebuild_instances_from_clouds(object_name)

    def blacklist_detection_latest_instance(self, object_name: str) -> bool:
        """Blacklist the latest detection instance for the class (backward compat)."""
        if object_name not in self._latest_detection_idx:
            return False
        idx = int(self._latest_detection_idx[object_name])
        return self.blacklist_detection_instance(object_name, idx)

    def blacklist_promoted_instance_near_xy(self, object_name: str, approx_xy: np.ndarray) -> bool:
        """Find and blacklist the nearest promoted instance around *approx_xy*."""
        insts = self.clouds.get(object_name, [])
        if not insts:
            return False
        centers = np.vstack([self._cloud_center_xy(inst) for inst in insts])
        d = np.linalg.norm(centers - np.asarray(approx_xy, dtype=np.float32).reshape(1, 2), axis=1)
        idx = int(np.argmin(d))
        inst = insts.pop(idx)
        self._blacklist_instances[object_name].append(inst)
        self._rebuild_instances_from_clouds(object_name)
        return True

    def blacklist_detection_instance(self, object_name: str, instance_idx: int) -> bool:
        """Move a specific detection instance to the instance-blacklist."""
        lst = self.detection_cloud.get(object_name, [])
        if instance_idx is None or instance_idx < 0 or instance_idx >= len(lst):
            return False
        inst = lst.pop(instance_idx)
        try:
            self._detection_frame_idx[object_name].pop(instance_idx)
        except Exception:
            pass
        try:
            self._promoted_match_for_detection[object_name].pop(instance_idx)
        except Exception:
            pass
        self._blacklist_instances[object_name].append(inst)
        self._adjust_latest_after_pop(object_name, instance_idx)
        return True

    # ------------------------------------------------------------------
    # Multi-signal candidate scoring
    # ------------------------------------------------------------------

    def record_candidate_attr_score(
        self, object_name: str, approx_xy: np.ndarray, score_sum: float
    ) -> None:
        if not np.isfinite(score_sum):
            return
        center = self.nearest_candidate_center(object_name, approx_xy)
        key = self._center_key(center)
        old = self._candidate_scores[object_name].get(key, (center, 0.0))[1]
        if score_sum > old:
            self._candidate_scores[object_name][key] = (center, float(score_sum))

    def record_candidate_ctx_score(
        self, object_name: str, approx_xy: np.ndarray, score: float
    ) -> None:
        if not np.isfinite(score):
            return
        center = self.nearest_candidate_center(object_name, approx_xy)
        key = self._center_key(center)
        old = self._candidate_ctx_scores[object_name].get(key, (center, 0.0))[1]
        if score > old:
            self._candidate_ctx_scores[object_name][key] = (center, float(score))

    def record_candidate_ctxcount_score(
        self, object_name: str, approx_xy: np.ndarray, count: float
    ) -> None:
        if not np.isfinite(count):
            return
        center = self.nearest_candidate_center(object_name, approx_xy)
        key = self._center_key(center)
        old = self._candidate_ctxcount_scores[object_name].get(key, (center, 0.0))[1]
        if count > old:
            self._candidate_ctxcount_scores[object_name][key] = (center, float(count))

    def record_candidate_rel_score(
        self, object_name: str, approx_xy: np.ndarray, count: float
    ) -> None:
        if not np.isfinite(count):
            return
        center = self.nearest_candidate_center(object_name, approx_xy)
        key = self._center_key(center)
        old = self._candidate_rel_scores[object_name].get(key, (center, 0.0))[1]
        if count > old:
            self._candidate_rel_scores[object_name][key] = (center, float(count))

    # ------------------------------------------------------------------
    # Score queries
    # ------------------------------------------------------------------

    def has_scored_candidates(self, object_name: str) -> bool:
        return any((
            len(self._candidate_scores.get(object_name, {})) > 0,
            len(self._candidate_ctx_scores.get(object_name, {})) > 0,
            len(self._candidate_ctxcount_scores.get(object_name, {})) > 0,
            len(self._candidate_rel_scores.get(object_name, {})) > 0,
        ))

    def best_scored_candidate(self, object_name: str) -> Union[np.ndarray, None]:
        tab_attr = self._candidate_scores.get(object_name, {})
        tab_ctx = self._candidate_ctx_scores.get(object_name, {})
        tab_ctxn = self._candidate_ctxcount_scores.get(object_name, {})
        tab_rel = self._candidate_rel_scores.get(object_name, {})
        if not tab_attr and not tab_ctx and not tab_ctxn and not tab_rel:
            return None

        keys = set().union(tab_attr.keys(), tab_ctx.keys(), tab_ctxn.keys(), tab_rel.keys())

        from vlfm.config import load_config
        _sc_cfg = load_config().get("scoring", {})
        w_attr = _sc_cfg.get("cand_w_attr", 1.0)
        w_ctx = _sc_cfg.get("cand_w_ctx", 1.0)
        w_ctxn = _sc_cfg.get("cand_w_ctxcount", 1.0)
        w_rel = _sc_cfg.get("cand_w_rel", 1.0)

        def _total(k: str) -> float:
            a = tab_attr.get(k, (None, 0.0))[1]
            c = tab_ctx.get(k, (None, 0.0))[1]
            n = tab_ctxn.get(k, (None, 0.0))[1]
            r = tab_rel.get(k, (None, 0.0))[1]
            return float(w_attr * a + w_ctx * c + w_ctxn * n + w_rel * r)

        best_key = max(keys, key=_total)
        return (
            tab_attr.get(best_key)
            or tab_ctx.get(best_key)
            or tab_ctxn.get(best_key)
            or tab_rel.get(best_key)
        )[0]

    # ------------------------------------------------------------------
    # Promoted relctx distance sums
    # ------------------------------------------------------------------

    def record_promoted_relctx_distance_sum(
        self, object_name: str, prom_index: int, sum_dist: float
    ) -> None:
        if not np.isfinite(sum_dist):
            return
        tab = self._promoted_relctx_sums[object_name]
        old = tab.get(int(prom_index), float("inf"))
        if float(sum_dist) < float(old):
            tab[int(prom_index)] = float(sum_dist)

    def best_promoted_index_by_relctx_sum(self, object_name: str) -> Union[None, int]:
        tab = self._promoted_relctx_sums.get(object_name, {})
        if not tab:
            return None
        return int(min(tab.keys(), key=lambda k: tab[k]))
