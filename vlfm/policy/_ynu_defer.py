# YNU defer and attribute defer mixin for BaseObjectNavPolicy

import base64
from typing import Optional, Set, Tuple

import cv2
import numpy as np


class YNUDeferMixin:
    """Yes/No/Unknown deferred verification and attribute scoring methods."""

    # ---- YNU defer ----

    def _ynu_start_defer(self, window: Optional[int] = None) -> None:
        self._ynu_defer_active = True
        if window is not None:
            self._ynu_defer_window = int(window)
        self._ynu_defer_start_step = int(self._num_steps)
        self._ynu_buffer = []

    def _ynu_collect_any(self, rgb: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        if not self._ynu_defer_active:
            return
        if mask is None or np.count_nonzero(mask) == 0:
            return
        try:
            clip = float(self._clip2itm.cosine(rgb, self.t_full))
        except Exception:
            clip = -1.0
        entry = {"rgb": rgb.copy(), "mask": (mask.copy() if mask is not None else None), "clip": float(clip)}
        self._ynu_buffer.append(entry)
        if len(self._ynu_buffer) > max(2, int(self._ynu_defer_window)):
            self._ynu_buffer = self._ynu_buffer[-int(self._ynu_defer_window):]

    def _ynu_ready(self) -> bool:
        return self._ynu_defer_active and (int(self._num_steps) - int(self._ynu_defer_start_step) >= int(self._ynu_defer_window))

    def _ynu_best_image_url(self) -> Optional[Tuple[str, bool]]:
        if not self._ynu_buffer:
            return None
        with_mask = [e for e in self._ynu_buffer if e.get("mask") is not None]
        cand = with_mask if len(with_mask) > 0 else list(self._ynu_buffer)
        best = max(cand, key=lambda e: float(e.get("clip", -1.0)))
        rgb = best["rgb"]
        mask = best.get("mask", None)
        if mask is not None and mask.shape[:2] == rgb.shape[:2]:
            url = self._vlm.make_masked_image_url(rgb, mask)
            return (url, True)
        _, buf = cv2.imencode(".jpg", rgb)
        b64 = base64.b64encode(buf).decode("utf-8")
        return (f"data:image/jpeg;base64,{b64}", False)

    def _ynu_try_finalize_deferred(self) -> None:
        """On window expiry, retry VLM once (Yes/No/Unknown rules preserved)."""
        if not self._ynu_ready():
            return
        best = self._ynu_best_image_url()
        if best is None:
            self._ynu_defer_active = False
            self._ynu_buffer = []
            return
        image_url, has_roi = best

        # (1) Existence re-check
        exist_prompt = (
            f"Is the {'outlined object ' if has_roi else ''}an instance of the category '{self._target_object}'?"
        )
        exist_ans = self._vlm.decision_ynu(exist_prompt, image_url)
        if exist_ans == "no":
            try:
                ok = self._object_map.blacklist_detection_latest_instance(self._target_object)
                if not ok and self._object_map.last_target_coord is not None:
                    self._object_map.blacklist_region(self._target_object, self._object_map.last_target_coord, self._blacklist_radius_m)
            except Exception:
                pass
            self._ynu_defer_active = False
            self._ynu_buffer = []
            return
        if exist_ans == "unknown":
            self._ynu_defer_active = False
            self._ynu_buffer = []
            try:
                approx_xy = self._object_map.last_target_coord
                if approx_xy is not None:
                    fts = self._observations_cache.get("frontier_sensor", np.array([]))
                    if isinstance(fts, np.ndarray) and fts.size > 0:
                        j = int(np.argmin(np.linalg.norm(fts - np.asarray(approx_xy).reshape(1, 2), axis=1)))
                        self._far_visit_goal = fts[j].astype(np.float32)
            except Exception:
                pass
            return

        # (2) Attribute re-check (only when existence is Yes)
        attr_dec = {}
        try:
            cats = set(getattr(self, "_pending_attr_categories", set()) or set())
            ask_cats = cats - set(getattr(self, "_ynu_passed_cats", set()))
            # Check cache for previously decided attributes
            _ynu_st = self._inst_state_store.get(self._active_inst_key, {}) if self._active_inst_key else {}
            _ynu_cached = dict(_ynu_st.get("attr_decisions", {}))
            _ynu_cached_cats = {c for c, d in _ynu_cached.items() if d in ("yes", "no")}
            _ynu_all_cats = {q.get("attribute") for q in (self._attr_questions or []) if q.get("attribute") in ("color", "material", "shape")}
            _ynu_uncached = _ynu_all_cats - _ynu_cached_cats - set(getattr(self, "_ynu_passed_cats", set()))
            if len(_ynu_uncached) == 0 and len(_ynu_cached_cats) > 0:
                attr_dec = dict(_ynu_cached)
                attr_scores = {}
            else:
                attr_dec, attr_scores = self._vlm.attribute_scores_and_decisions(
                    image_url=image_url,
                    attr_questions=self._attr_questions,
                    categories=_ynu_uncached if len(_ynu_uncached) < len(_ynu_all_cats) else None,
                    ynu_passed_cats=getattr(self, "_ynu_passed_cats", set())
                )
                # Merge cached decisions
                for c, d in _ynu_cached.items():
                    if c not in attr_dec:
                        attr_dec[c] = d
                # Update cache with new definitive results
                if self._active_inst_key:
                    for c, d in attr_dec.items():
                        if d in ("yes", "no"):
                            _ynu_st.setdefault("attr_decisions", {})[c] = d
                    self._inst_state_store[self._active_inst_key] = _ynu_st
            # Add definitive-Yes categories to ynu_passed_cats immediately
            for c, d in attr_dec.items():
                if d == "yes":
                    self._ynu_passed_cats.add(c)
        except Exception:
            attr_dec, attr_scores = {}, {}
        if any(v == "no" for v in attr_dec.values()):
            try:
                ok = self._object_map.blacklist_detection_latest_instance(self._target_object)
                if not ok and self._object_map.last_target_coord is not None:
                    self._object_map.blacklist_region(self._target_object, self._object_map.last_target_coord, self._blacklist_radius_m)
            except Exception:
                pass
            self._ynu_unknown_counts = {}
            self._ynu_passed_cats = set()
            self._ynu_defer_active = False
            self._ynu_buffer = []
            return
        unknown_all = {k for k, v in attr_dec.items() if v == "unknown"} - set(getattr(self, "_ynu_passed_cats", set()))
        unknown_cats = self._ynu_bump_and_filter_unknown(unknown_all)
        try:
            ok, final_pidx = self._object_map.commit_latest_detection_to_clouds(self._target_object, set_last_coord=True)
            if ok:
                det_idx = self._object_map._latest_detection_idx.get(self._target_object, None)
                if det_idx is not None and self._active_inst_key is not None:
                    self._inst_bind_after_commit(self._target_object, int(det_idx), self._active_inst_key, int(final_pidx))
                try:
                    prom_center = self._object_map.promoted_center(self._target_object, int(final_pidx))
                    if prom_center is not None:
                        yes_cats = [k for k, v in attr_dec.items() if v == "yes"]
                        if len(yes_cats) > 0:
                            s_total = float(sum(float(attr_scores.get(k, 0.0)) for k in yes_cats))
                            self._object_map.record_candidate_attr_score(
                                self._target_object, np.asarray(prom_center, np.float32).reshape(2), s_total
                            )
                except Exception:
                    pass
                try:
                    det_idx = self._object_map._latest_detection_idx.get(self._target_object, None)
                    _, det_center = (self._object_map.detection_center_key(self._target_object, int(det_idx))
                                     if det_idx is not None else (None, None))
                except Exception:
                    det_center = None
                self._record_relctx_distance_for_promoted(final_pidx, center_hint=det_center)
                self._pending_attr_categories = set(unknown_cats)
                if len(self._pending_attr_categories) == 0:
                    self._pending_attr_categories = set()
        except Exception:
            pass
        self._ynu_defer_active = False
        self._ynu_buffer = []
        # Only clear attr defer if all attributes have passed
        if len(getattr(self, "_pending_attr_categories", set()) or set()) == 0:
            self._attr_defer_active = False
            self._attr_buffer = []

    def _ynu_bump_and_filter_unknown(self, cats: Set[str]) -> Set[str]:
        """
        Bump unknown count per category. If count >= threshold, mark as passed.
        Returns categories still needing re-verification.
        """
        remain: Set[str] = set()
        for c in (cats or set()):
            if not isinstance(c, str):
                continue
            cnt = int(self._ynu_unknown_counts.get(c, 0)) + 1
            self._ynu_unknown_counts[c] = cnt
            if cnt >= int(self._ynu_unknown_pass_after):
                self._ynu_passed_cats.add(c)
            else:
                remain.add(c)
        return remain

    # ---- Attribute defer ----

    def _attr_start_defer(self, window: Optional[int] = None) -> None:
        self._attr_defer_active = True
        if window is not None:
            self._attr_defer_window = int(window)
        self._attr_defer_start_step = int(self._num_steps)
        self._attr_buffer = []

    def _attr_collect_any(self, rgb: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        if not self._attr_defer_active:
            return
        try:
            clip = float(self._clip2itm.cosine(rgb, self.t_full))
        except Exception:
            clip = -1.0
        self._attr_buffer.append({"rgb": rgb.copy(), "mask": (mask.copy() if mask is not None else None), "clip": float(clip)})
        if len(self._attr_buffer) > max(2, int(self._attr_defer_window)):
            self._attr_buffer = self._attr_buffer[-int(self._attr_defer_window):]

    def _attr_ready(self) -> bool:
        return self._attr_defer_active and (int(self._num_steps) - int(self._attr_defer_start_step) >= int(self._attr_defer_window))

    def _attr_best_image_url(self) -> Optional[Tuple[str, bool]]:
        if not self._attr_buffer:
            return None
        with_mask = [e for e in self._attr_buffer if e.get("mask") is not None]
        cand = with_mask if len(with_mask) > 0 else list(self._attr_buffer)
        best = max(cand, key=lambda e: float(e.get("clip", -1.0)))
        rgb = best["rgb"]
        mask = best.get("mask", None)
        if mask is not None and mask.shape[:2] == rgb.shape[:2]:
            url = self._vlm.make_masked_image_url(rgb, mask)
            return (url, True)
        _, buf = cv2.imencode(".jpg", rgb)
        b64 = base64.b64encode(buf).decode("utf-8")
        return (f"data:image/jpeg;base64,{b64}", False)

    def _attr_try_finalize_deferred(self) -> None:
        """On window expiry, re-verify only pending attribute categories."""
        if not self._attr_ready():
            return
        cats_all = set(getattr(self, "_pending_attr_categories", set()) or set())
        cats_ask = cats_all - set(getattr(self, "_ynu_passed_cats", set()))
        if len(cats_ask) == 0:
            self._attr_defer_active = False
            self._attr_buffer = []
            return
        best = self._attr_best_image_url()
        if best is None:
            self._attr_defer_active = False
            self._attr_buffer = []
            return
        image_url, _ = best
        # Check cache for previously decided attributes
        _attr_st = self._inst_state_store.get(self._active_inst_key, {}) if self._active_inst_key else {}
        _attr_cached = dict(_attr_st.get("attr_decisions", {}))
        _attr_cached_cats = {c for c, d in _attr_cached.items() if d in ("yes", "no")}
        _attr_uncached = cats_ask - _attr_cached_cats
        try:
            if len(_attr_uncached) == 0 and len(_attr_cached_cats) > 0:
                attr_dec = {c: _attr_cached[c] for c in cats_ask if c in _attr_cached}
                attr_scores = {}
            else:
                attr_dec, attr_scores = self._vlm.attribute_scores_and_decisions(
                    image_url=image_url,
                    attr_questions=self._attr_questions,
                    categories=_attr_uncached if len(_attr_uncached) < len(cats_ask) else cats_ask,
                    ynu_passed_cats=self._ynu_passed_cats,
                )
                # Merge cached decisions
                for c, d in _attr_cached.items():
                    if c in cats_ask and c not in attr_dec:
                        attr_dec[c] = d
                # Update cache with new definitive results
                if self._active_inst_key:
                    for c, d in attr_dec.items():
                        if d in ("yes", "no"):
                            _attr_st.setdefault("attr_decisions", {})[c] = d
                    self._inst_state_store[self._active_inst_key] = _attr_st
            # Add definitive-Yes categories to ynu_passed_cats immediately
            for c, d in attr_dec.items():
                if d == "yes":
                    self._ynu_passed_cats.add(c)
        except Exception:
            attr_dec, attr_scores = {}, {}
        if any(v == "no" for v in attr_dec.values()):
            try:
                ok = self._object_map.blacklist_detection_latest_instance(self._target_object)
                if not ok and self._object_map.last_target_coord is not None:
                    self._object_map.blacklist_region(self._target_object, self._object_map.last_target_coord, self._blacklist_radius_m)
            except Exception:
                pass
            self._ynu_unknown_counts = {}
            self._ynu_passed_cats = set()
            self._pending_attr_categories = set()
            self._attr_defer_active = False
            self._attr_buffer = []
            return
        unknown_all = {k for k, v in attr_dec.items() if v == "unknown"} - set(getattr(self, "_ynu_passed_cats", set()))
        unknown_pending = self._ynu_bump_and_filter_unknown(unknown_all)
        try:
            ok, final_pidx = self._object_map.commit_latest_detection_to_clouds(self._target_object, set_last_coord=True)
            if ok and self._active_inst_key is not None:
                det_idx = self._object_map._latest_detection_idx.get(self._target_object, None)
                if det_idx is not None:
                    self._inst_bind_after_commit(self._target_object, int(det_idx), self._active_inst_key, int(final_pidx))
        except Exception:
            pass
        self._pending_attr_categories = set(unknown_pending)
        self._attr_defer_active = False
        self._attr_buffer = []
