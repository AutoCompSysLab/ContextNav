# Object map update mixin for BaseObjectNavPolicy

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np

from vlfm.utils.geometry_utils import get_fov
from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.grounding_dino import ObjectDetections


class ObjectMapUpdateMixin:
    """Logic for updating the object point cloud map from detections each step."""

    def _update_object_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections:
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.
        """
        detections = self._get_object_detections(rgb)
        h, w = rgb.shape[:2]
        self._object_masks = np.zeros((h, w), dtype=np.uint8)
        self._object_masks_vis = np.zeros((h, w), dtype=np.uint8)

        if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0:
            depth = self._infer_depth(rgb, min_depth, max_depth)
            obs = list(self._observations_cache["object_map_rgbd"][0])
            obs[1] = depth
            self._observations_cache["object_map_rgbd"][0] = tuple(obs)

        frame_inst_by_class: Dict[str, Set[int]] = defaultdict(set)
        matched_prom_by_instance: Dict[Tuple[str, int], int] = {}
        inst_masks: Dict[Tuple[str, int], np.ndarray] = {}
        inst_is_new: Dict[Tuple[str, int], bool] = {}
        newly_committed: Set[str] = set()
        frame_id = int(self._num_steps)
        ig_high_set = set(self._ig_high_categories or [])
        robot_xy = self._observations_cache["robot_xy"]

        for idx in range(len(detections.logits)):
            bbox_denorm = detections.boxes[idx] * np.array([w, h, w, h])
            object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())
            if object_mask is None or np.count_nonzero(object_mask) == 0:
                continue

            self._object_masks = np.zeros_like(self._object_masks)
            self._object_masks[object_mask > 0] = 1
            self._object_masks_vis[object_mask > 0] = 1

            cls_name = detections.phrases[idx]

            # Max-depth mask handling
            md_th = self._cfg_maxdepth_norm_th
            md_ratio_th = self._cfg_maxdepth_skip_ratio
            dm = depth[object_mask > 0]
            is_maxdepth_mask = (dm.size > 0) and (float(np.mean(dm >= md_th)) >= md_ratio_th)
            is_target = (cls_name == self._target_object)
            if is_maxdepth_mask:
                continue

            is_new, det_idx, prom_idx = self._object_map.update_map(
                cls_name, depth, object_mask, tf_camera_to_episodic, min_depth, max_depth, fx, fy,
                is_target_object=is_target,
                frame_id=frame_id,
            )
            if det_idx < 0:
                continue
            inst_is_new[(cls_name, int(det_idx))] = bool(is_new)
            matched_prom_by_instance[(cls_name, int(det_idx))] = int(prom_idx)
            frame_inst_by_class[cls_name].add(int(det_idx))
            k = (cls_name, int(det_idx))
            mk = (object_mask > 0).astype(np.uint8)
            if k not in inst_masks:
                inst_masks[k] = mk
            else:
                inst_masks[k] |= mk
        # --- Phase-B: per-instance processing after frame ---
        target = self._target_object

        # (1) Non-target COCO: commit without verification
        for cname, id_set in list(frame_inst_by_class.items()):
            if cname == target or (cname not in COCO_CLASSES):
                continue
            for det_idx in sorted(list(id_set), reverse=True):
                try:
                    ok, final_pidx = self._object_map.commit_detection_instance_to_clouds(cname, det_idx, set_last_coord=False)
                    if ok:
                        newly_committed.add(cname)
                except Exception as e:
                    print(f"[MAP] commit_detection_instance_to_clouds({cname},{det_idx}) failed: {e}")

        # (2) Non-target non-COCO: verify existence then commit/blacklist
        noncoco_inst = [(c, i) for (c, ids) in frame_inst_by_class.items() if (c != target and c not in COCO_CLASSES) for i in ids]
        if len(noncoco_inst) > 0 and self._verify_with_vlm:
            masks_by_instance = {k: inst_masks.get(k, None) for k in noncoco_inst}
            th = self._cfg_vlm_decision_threshold
            try:
                res = self._vlm.verify_non_coco_instances_exists(rgb, noncoco_inst, threshold=th, masks_by_instance=masks_by_instance)
            except Exception as e:
                print(f"[VERIFY] non-COCO instance-batch error: {e}")
                res = {k: False for k in noncoco_inst}
            for (cname, det_idx), ok in res.items():
                try:
                    if ok:
                        okc, final_pidx = self._object_map.commit_detection_instance_to_clouds(cname, det_idx, set_last_coord=False)
                        if okc:
                            newly_committed.add(cname)
                    else:
                        self._object_map.blacklist_detection_instance(cname, det_idx)
                except Exception as e:
                    print(f"[MAP] finalize non-COCO({cname},{det_idx}) error: {e}")

        # (3) Target instances: per-instance existence/attribute verification
        self._process_target_instances(
            target, frame_inst_by_class, inst_masks, inst_is_new,
            matched_prom_by_instance, newly_committed, rgb, depth,
            robot_xy, ig_high_set,
        )

        # Update explored region
        try:
            cone_fov = get_fov(fx, depth.shape[1])
            self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov)
        except Exception:
            pass

        # Finalize deferred instances whose windows matured
        try:
            for ik, st in list(self._inst_state_store.items()):
                if st.get("ynu_defer_active") and (self._num_steps - int(st.get("ynu_defer_start_step", 0)) >= int(self._ynu_defer_window)):
                    self._inst_pull(ik)
                    try:
                        self._ynu_try_finalize_deferred()
                    except Exception as e:
                        print(f"[YNU] finalize({ik}) error: {e}")
                    self._inst_push(ik)
                if st.get("attr_defer_active") and (self._num_steps - int(st.get("attr_defer_start_step", 0)) >= int(self._attr_defer_window)):
                    self._inst_pull(ik)
                    try:
                        self._attr_try_finalize_deferred()
                    except Exception as e:
                        print(f"[ATTR] finalize({ik}) error: {e}")
                    self._inst_push(ik)
        except Exception as e:
            print(f"[DEFER] finalize sweep error: {e}")

        # Opportunistic commit when new context is committed
        try:
            new_ctx = set(self._context_categories).intersection(newly_committed)
            if len(new_ctx) > 0:
                self._try_commit_targets_with_new_context(list(new_ctx))
        except Exception as e:
            print(f"[CTX->TGT] opportunistic commit error: {e}")
        return detections

    def _process_target_instances(
        self,
        target: str,
        frame_inst_by_class: Dict[str, Set[int]],
        inst_masks: Dict[Tuple[str, int], np.ndarray],
        inst_is_new: Dict[Tuple[str, int], bool],
        matched_prom_by_instance: Dict[Tuple[str, int], int],
        newly_committed: Set[str],
        rgb: np.ndarray,
        depth: np.ndarray,
        robot_xy: np.ndarray,
        ig_high_set: Set[str],
    ) -> None:
        """Process target object instances: existence check, attribute check, context check."""
        if target not in frame_inst_by_class:
            return

        enable_vlm_exist_check = self._cfg_enable_vlm_exist_check

        for det_idx in sorted(list(frame_inst_by_class[target]), reverse=True):
            mask = inst_masks.get((target, det_idx), None)
            prom_idx = int(matched_prom_by_instance.get((target, det_idx), -1))
            store_key = self._inst__store_key_for(target, det_idx, prom_idx)
            st = self._inst__ensure(store_key) if store_key is not None else {}
            self._far_visit_goal = None

            try:
                _is_new = bool(inst_is_new.get((target, int(det_idx)), True))
            except Exception:
                _is_new = True
            if not _is_new:
                will_create_new_promoted = (prom_idx < 0)
                if not will_create_new_promoted and st.get("passed_attrs", False):
                    ok, final_pidx = self._object_map.commit_detection_instance_to_clouds(
                        target, det_idx, set_last_coord=True
                    )
                    if ok:
                        self._inst_bind_after_commit(target, det_idx, store_key, final_pidx)
                    try:
                        st["last_passed_step"] = int(self._num_steps)
                        self._inst_state_store[store_key] = st
                    except Exception:
                        pass
                    continue

            if getattr(self, "_attr_defer_active", False):
                ok, final_pidx = self._object_map.commit_detection_instance_to_clouds(target, det_idx, set_last_coord=True)
                if ok:
                    self._inst_bind_after_commit(target, det_idx, store_key, final_pidx)
                try:
                    if mask is not None:
                        if self._ynu_defer_active:
                            self._ynu_collect_any(rgb, mask)
                        if self._attr_defer_active:
                            self._attr_collect_any(rgb, mask)
                    else:
                        if self._ynu_defer_active:
                            self._ynu_collect_any(rgb, None)
                        if self._attr_defer_active:
                            self._attr_collect_any(rgb, None)
                except Exception:
                    pass
                if store_key is not None:
                    self._inst_push(store_key)
                continue
            okc = False

            # (a) Existence check
            need_exist_check = not bool(st.get("passed_exist", False))
            if need_exist_check:
                if enable_vlm_exist_check:
                    try:
                        url = self._vlm.make_masked_image_url(rgb, mask) if mask is not None else None
                        exist_ans = self._vlm.decision_ynu(
                            f"Is the outlined object an instance of the category '{target}'?",
                            url if url is not None else self._vlm.make_masked_image_url(rgb, np.zeros_like(depth, dtype=np.uint8))
                        )
                    except Exception as e:
                        print(f"[VERIFY] target exist YNU error: {e}")
                        exist_ans = "unknown"
                else:
                    exist_ans = "yes"
            else:
                exist_ans = "yes"

            if exist_ans == "yes":
                st["passed_exist"] = True
                self._inst_state_store[store_key] = st

            elif exist_ans == "unknown":
                self._ynu_start_defer(window=self._ynu_defer_window)
                try:
                    if mask is not None and np.count_nonzero(mask) > 0:
                        self._ynu_collect_any(rgb, mask)
                except Exception:
                    pass
                try:
                    _, det_center = self._object_map.detection_center_key(target, det_idx)
                    fts = self._observations_cache.get("frontier_sensor", np.array([]))
                    if det_center is not None and isinstance(fts, np.ndarray) and fts.size > 0:
                        robot_xy = self._observations_cache["robot_xy"]
                        cand = [ft for ft in fts if not self._wall_line_blocked(robot_xy, ft)]
                        F = np.array(cand) if len(cand) > 0 else fts
                        j = int(np.argmin(np.linalg.norm(F - det_center.reshape(1, 2), axis=1)))
                        self._far_visit_goal = F[j].astype(np.float32)
                except Exception:
                    pass
                if store_key is not None:
                    self._inst_push(store_key)
                continue

            elif exist_ans == "no":
                try:
                    self._object_map.blacklist_detection_instance(target, det_idx)
                    self._det2store_key.pop((target, int(det_idx)), None)
                except Exception:
                    pass
                self._ynu_unknown_counts = {}
                self._ynu_passed_cats = set()
                if store_key is not None:
                    self._inst_push(store_key)
                continue
            else:
                continue

            # (b) Attribute verification
            attr_dec, attr_scores = {}, {}
            if self._attr_questions is not None:
                # Check cache for previously decided attributes
                cached_decisions = dict(st.get("attr_decisions", {}))
                cached_cats = {c for c, d in cached_decisions.items() if d in ("yes", "no")}
                all_cats = {q.get("attribute") for q in self._attr_questions if q.get("attribute") in ("color", "material", "shape")}
                uncached_cats = all_cats - cached_cats - getattr(self, "_ynu_passed_cats", set())

                if len(uncached_cats) == 0 and len(cached_cats) > 0:
                    # All categories already cached, skip VLM call
                    attr_dec = dict(cached_decisions)
                    attr_scores = {}
                else:
                    try:
                        url = self._vlm.make_masked_image_url(rgb, mask) if mask is not None else None
                        attr_dec, attr_scores = self._vlm.attribute_scores_and_decisions(
                            image_url=url,
                            attr_questions=self._attr_questions,
                            categories=uncached_cats if len(uncached_cats) < len(all_cats) else None,
                            ynu_passed_cats=getattr(self, "_ynu_passed_cats", set())
                        )
                    except Exception as e:
                        print(f"[VERIFY] attr YNU error: {e}")
                        attr_dec, attr_scores = {}, {}
                    # Merge cached decisions into results
                    for c, d in cached_decisions.items():
                        if c not in attr_dec:
                            attr_dec[c] = d
                    # Update cache with new definitive results
                    for c, d in attr_dec.items():
                        if d in ("yes", "no"):
                            st.setdefault("attr_decisions", {})[c] = d
                    self._inst_state_store[store_key] = st
                # Add definitive-Yes categories to ynu_passed_cats immediately
                for c, d in attr_dec.items():
                    if d == "yes":
                        self._ynu_passed_cats.add(c)
                if any(v == "no" for v in attr_dec.values()):
                    try:
                        _, det_center = self._object_map.detection_center_key(target, det_idx)
                        if det_center is not None:
                            yes_cats = [k for k, v in attr_dec.items() if v == "yes"]
                            s_total = float(sum(float(attr_scores.get(k, 0.0)) for k in yes_cats))
                            neigh = self._neighbor_context_instances(det_center, radius_m=3.0)
                            neighbor_cnt = sum(len(v) for v in neigh.values())
                            rels = self._effective_relations_around_target(det_center, neigh, include_context_only=True)
                            sat_cnt = 0
                            if len(rels) > 0:
                                assign = {self._target_object: det_center}
                                for c, pts in neigh.items():
                                    if len(pts) > 0:
                                        assign[c] = pts[0]
                                ok, _ = self._find_joint_viewpoint_for_relations(rels, assign)
                                sat_cnt = len(rels) if ok else 0
                            self._object_map.record_candidate_attr_score(target, det_center, s_total)
                            self._object_map.record_candidate_ctxcount_score(target, det_center, len(neigh))
                            self._object_map.record_candidate_rel_score(target, det_center, float(sat_cnt))
                    except Exception:
                        pass
                    try:
                        self._object_map.blacklist_detection_instance(target, det_idx)
                        self._det2store_key.pop((target, int(det_idx)), None)
                    except Exception:
                        pass
                    self._ynu_unknown_counts = {}
                    self._ynu_passed_cats = set()
                    if store_key is not None:
                        self._inst_push(store_key)
                    continue

                # (c) Save attribute state
                unknown_cats_all = {k for k, v in attr_dec.items() if v == "unknown"} - getattr(self, "_ynu_passed_cats", set())
                unknown_pending = self._ynu_bump_and_filter_unknown(unknown_cats_all)
                self._pending_attr_categories = set(unknown_pending)
                try:
                    if store_key is not None:
                        st = self._inst__ensure(store_key)
                        st["passed_attrs"] = (len(self._pending_attr_categories) == 0) and (all(v != "no" for v in attr_dec.values()))
                        if st["passed_attrs"]:
                            st["last_passed_step"] = int(self._num_steps)
                        self._inst_state_store[store_key] = st
                except Exception:
                    pass
                if len(self._pending_attr_categories) > 0:
                    self._attr_start_defer(window=self._attr_defer_window)
                    try:
                        self._attr_collect_any(rgb, mask)
                    except Exception:
                        pass
            else:
                st = self._inst__ensure(store_key)
                st["passed_attrs"] = True
                self._inst_state_store[store_key] = st
                try:
                    _, det_center = self._object_map.detection_center_key(target, det_idx)
                    if det_center is not None and isinstance(self._context_description, str) and len(self._context_description) > 0:
                        s_ctx = float(self._clip2itm.cosine(rgb, self._context_description))
                        if np.isfinite(s_ctx):
                            self._object_map.record_candidate_ctx_score(
                                target, det_center, max(0.0, s_ctx)
                            )
                except Exception:
                    pass

            # (d) Context/relation check, then commit if possible
            try:
                _, det_center = self._object_map.detection_center_key(target, det_idx)
                can_commit = False
                neighbor_cnt, rel_cnt, satisfied_cnt = 0, 0, 0
                if det_center is not None:
                    neigh = self._neighbor_context_instances(det_center, radius_m=3.0)
                    neighbor_cnt = sum(len(v) for v in neigh.values())
                    if neighbor_cnt > 0:
                        rels = self._effective_relations_around_target(
                            det_center, neigh, include_context_only=True
                        )
                        rel_cnt = len(rels)
                        if rel_cnt == 0:
                            can_commit = True
                        else:
                            assign = {self._target_object: det_center}
                            for c, pts in neigh.items():
                                if len(pts) > 0:
                                    assign[c] = pts[0]
                            ok, _ = self._find_joint_viewpoint_for_relations(rels, assign)
                            can_commit = bool(ok)
                            if ok:
                                satisfied_cnt = rel_cnt

                try:
                    if det_center is not None:
                        yes_cats = [k for k, v in attr_dec.items() if v == "yes"]
                        s_total = 0.0
                        if len(yes_cats) > 0:
                            s_total = float(sum(float(attr_scores.get(k, 0.0)) for k in yes_cats))
                        self._object_map.record_candidate_attr_score(target, det_center, float(s_total))
                        self._object_map.record_candidate_ctxcount_score(target, det_center, len(neigh))
                        self._object_map.record_candidate_rel_score(target, det_center, float(satisfied_cnt))
                except Exception:
                    pass
                if can_commit:
                    okc, final_pidx = self._object_map.commit_detection_instance_to_clouds(target, det_idx, set_last_coord=True)
                    if okc:
                        self._inst_bind_after_commit(target, det_idx, store_key, final_pidx)
                        newly_committed.add(target)
                        try:
                            _, det_center = self._object_map.detection_center_key(target, det_idx)
                        except Exception:
                            det_center = None
                        self._record_relctx_distance_for_promoted(final_pidx, center_hint=det_center)
                else:
                    if det_center is not None and (len(self._context_categories) > 0):
                        st_local = self._inst__ensure(store_key)
                        already = bool(st_local.get("ctx_frontier_explored_once", False))
                        ft_same = self._choose_same_room_frontier(det_center)
                        if (not already) and (ft_same is not None) and (not self._frontier_lock_active):
                            self._lock_frontier(ft_same, reason="tgt_ctx_once", store_key=store_key)
                            st_local["ctx_frontier_explored_once"] = True
                            self._inst_state_store[store_key] = st_local
            except Exception as e:
                print(f"[COMMIT-DEFER] target commit via context failed: {e}")
            if store_key is not None:
                self._inst_push(store_key)

    def _try_commit_targets_with_new_context(self, ctx_classes: list) -> None:
        """After context is committed, try to promote target candidates that now have context support."""
        tgt = self._target_object
        lst = self._object_map.detection_cloud.get(tgt, [])
        if not isinstance(lst, list) or len(lst) == 0:
            return
        candidates = []
        for det_idx in range(len(lst)):
            prom_idx = self._object_map.promoted_match_index(tgt, det_idx)
            store_key = self._inst__store_key_for(tgt, det_idx, prom_idx)
            st = self._inst_state_store.get(store_key, {})
            if not (st.get("passed_exist", False) and st.get("passed_attrs", False)):
                continue
            _, c = self._object_map.detection_center_key(tgt, det_idx)
            if c is None:
                continue
            candidates.append((det_idx, c, store_key))
        if len(candidates) == 0:
            return

        ok_list = []
        for cname in ctx_classes:
            ctx_pts = self._object_map.list_candidate_centers(cname)
            for det_idx, c, store_key in candidates:
                for ctx_c in ctx_pts:
                    if not self._same_room_by_wall(c, ctx_c):
                        continue
                    neigh = {cname: [np.asarray(ctx_c, np.float32)]}
                    rels = self._effective_relations_around_target(c, neigh, include_context_only=False)
                    if len(rels) == 0:
                        ok_list.append((float(np.linalg.norm(c - ctx_c)), det_idx, store_key))
                    else:
                        assign = {self._target_object: c, cname: np.asarray(ctx_c, np.float32)}
                        ok, _ = self._find_joint_viewpoint_for_relations(rels, assign)
                        if ok:
                            ok_list.append((float(np.linalg.norm(c - ctx_c)), det_idx, store_key))
        if len(ok_list) == 0:
            return
        ok_list.sort(key=lambda t: t[0])
        _, det_idx, store_key = ok_list[0]
        okc, final_pidx = self._object_map.commit_detection_instance_to_clouds(tgt, det_idx, set_last_coord=True)
        if okc:
            self._inst_bind_after_commit(tgt, det_idx, store_key, final_pidx)
            _, c = self._object_map.detection_center_key(tgt, det_idx)
            self._record_relctx_distance_for_promoted(final_pidx, center_hint=c)
