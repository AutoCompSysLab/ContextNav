# Spatial reasoning mixin for BaseObjectNavPolicy

from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class SpatialReasoningMixin:
    """Relation checking, joint viewpoint search, and context-based instance selection."""

    def _relation_holds_at_pose(
        self,
        rtype: str,
        ref_xy: np.ndarray,
        tgt_xy: np.ndarray,
        vp_xy: np.ndarray,
        ref_z: float = None,
        tgt_z: float = None,
    ) -> bool:
        """
        Check whether a spatial relation holds from a given viewpoint.
        Observer yaw is set so that vp->ref is the forward direction.
        """
        p = np.asarray(vp_xy, np.float32).reshape(2)
        ref = np.asarray(ref_xy, np.float32).reshape(2)
        tgt = np.asarray(tgt_xy, np.float32).reshape(2)
        yaw = float(np.arctan2(ref[1] - p[1], ref[0] - p[0]))

        ca, sa = np.cos(-yaw), np.sin(-yaw)

        def rot(v):
            vx, vy = v[0] - p[0], v[1] - p[1]
            return np.array([vx * ca - vy * sa, vx * sa + vy * ca], np.float32)

        r = rot(ref)
        t = rot(tgt)
        eps_m = 0.15
        ang_eps = np.deg2rad(25.0)
        if rtype == "left":
            return (t[1] - r[1]) > eps_m
        if rtype == "right":
            return (r[1] - t[1]) > eps_m

        def bearing(u):
            return np.arctan2(u[1], u[0])

        if rtype == "front":
            return (abs(bearing(t) - bearing(r)) <= ang_eps) and (t[0] < r[0] - eps_m)
        if rtype == "behind":
            return (abs(bearing(t) - bearing(r)) <= ang_eps) and (t[0] > r[0] + eps_m)
        if rtype == "near":
            d = float(np.linalg.norm(ref - tgt))
            return d <= float(getattr(self, "_relation_max_distance_m", 2.0))
        if rtype == "below":
            if (ref_z is None) or (tgt_z is None):
                return True  # Cannot evaluate; treat as soft pass
            z_eps = 0.15
            return (tgt_z + z_eps) > (ref_z - 1e-6)
        # Unsupported relation types are treated as soft pass
        return True

    def _neighbor_context_instances(
        self,
        center_xy: np.ndarray,
        radius_m: float = 3.0,
        max_per_class: int = None,
    ) -> dict:
        """
        Collect context instances within radius and same room of the target center.
        Returns: {class_name: [np.ndarray(2,), ...], ...}
        """
        out = {}
        cats = set(self._context_categories or [])
        for cname in sorted(cats):
            centers = self._object_map.list_candidate_centers(cname)
            keep = []
            for c in centers:
                c = np.asarray(c, np.float32).reshape(2)
                if np.linalg.norm(c - center_xy) <= float(radius_m) and self._same_room_by_wall(center_xy, c):
                    keep.append(c)
            if len(keep) == 0:
                continue
            keep.sort(key=lambda p: float(np.linalg.norm(p - center_xy)))
            if isinstance(max_per_class, int) and max_per_class > 0:
                keep = keep[:max_per_class]
            out[cname] = keep
        return out

    def _effective_relations_around_target(
        self,
        tgt_center: np.ndarray,
        neighbors: dict,
        include_context_only: bool = True,
    ) -> list:
        """
        Filter text-extracted relations to only those whose objects exist in the current map.
        """
        rels = []
        neigh_classes = set(neighbors.keys())
        for r in (self._context_relations or []):
            ref, tgt, typ = r.get("ref"), r.get("tgt"), r.get("rtype")
            if ref is None or tgt is None or typ is None:
                continue
            if typ not in self._supported_rtypes:
                continue
            if self._target_object in (ref, tgt):
                other = tgt if ref == self._target_object else ref
                if other in neigh_classes:
                    rels.append({"ref": ref, "tgt": tgt, "rtype": typ})
            else:
                if include_context_only and (ref in neigh_classes) and (tgt in neigh_classes):
                    rels.append({"ref": ref, "tgt": tgt, "rtype": typ})
        return rels

    def _compute_relation_context_distance_sum(
        self,
        tgt_center: np.ndarray,
        radius_m: float = 3.0,
    ) -> float:
        """
        Sum distances from tgt_center to context instances that satisfy text-defined
        relations via joint viewpoint existence.
        """
        if not isinstance(tgt_center, np.ndarray):
            return float("inf")
        S = 0.0
        rels_all = list(self._context_relations or [])
        if len(rels_all) == 0:
            return float("inf")
        for r in rels_all:
            ref, tgt, rtype = r.get("ref"), r.get("tgt"), r.get("rtype")
            if ref is None or tgt is None or rtype is None:
                continue
            if self._target_object not in (ref, tgt):
                continue
            other_cls = (tgt if ref == self._target_object else ref)
            ctx_centers = self._object_map.list_candidate_centers(other_cls)
            if len(ctx_centers) == 0:
                continue
            for cxy in ctx_centers:
                cxy = np.asarray(cxy, np.float32).reshape(2)
                if not self._same_room_by_wall(tgt_center, cxy):
                    continue
                if rtype == "near":
                    if float(np.linalg.norm(tgt_center - cxy)) > float(getattr(self, "_relation_max_distance_m", 2.0)):
                        continue
                assign = {self._target_object: tgt_center, other_cls: cxy}
                ok, _vp = self._find_joint_viewpoint_for_relations([{"ref": ref, "tgt": tgt, "rtype": rtype}], assign)
                if ok:
                    S += float(np.linalg.norm(tgt_center - cxy))
        return S if np.isfinite(S) and S > 0 else float("inf")

    # Alias used internally
    _relctx_distance_sum = _compute_relation_context_distance_sum

    def _candidate_viewpoints_for_group(
        self,
        centers: list,
        rel_pairs: list,
    ) -> list:
        """
        Generate candidate viewpoints to observe multiple objects simultaneously:
        ring samples around midpoints of relation pairs and overall centroid.
        """
        cand = []
        radii = [0.8, 1.2, 1.6, 2.0]
        n_ang = 24
        for (a, b) in rel_pairs:
            mid = 0.5 * (np.asarray(a, np.float32) + np.asarray(b, np.float32))
            for r in radii:
                for k in range(n_ang):
                    th = (2 * np.pi * k) / n_ang
                    cand.append(mid + np.array([r * np.cos(th), r * np.sin(th)], np.float32))
        if len(centers) > 0:
            cen = np.mean(np.vstack(centers), axis=0).astype(np.float32)
            for r in radii:
                for k in range(n_ang):
                    th = (2 * np.pi * k) / n_ang
                    cand.append(cen + np.array([r * np.cos(th), r * np.sin(th)], np.float32))
        key_fn = lambda p: (round(float(p[0]), 1), round(float(p[1]), 1))
        uniq = {}
        for p in cand:
            uniq.setdefault(key_fn(p), p)
        return list(uniq.values())

    def _find_joint_viewpoint_for_relations(
        self,
        relations: list,
        assign: dict,
    ):
        """
        Search for a single viewpoint from which all relations are simultaneously satisfied.
        """
        centers = [assign[r["ref"]] for r in relations] + [assign[r["tgt"]] for r in relations]
        pair_list = [(assign[r["ref"]], assign[r["tgt"]]) for r in relations]
        cands = self._candidate_viewpoints_for_group(centers, pair_list)

        def _z_at(cls_name: str, xy: np.ndarray):
            def _mean_z(inst: np.ndarray):
                if inst is None or inst.shape[0] == 0:
                    return None
                pts = inst[inst[:, -1] == 1]
                pts = pts if pts.shape[0] > 0 else inst
                return float(np.mean(pts[:, 2])) if pts.shape[0] > 0 else None

            insts = self._object_map.clouds.get(cls_name, [])
            best_d, best_z = 1e9, None
            for cl in insts:
                if not isinstance(cl, np.ndarray) or cl.shape[0] == 0:
                    continue
                cxy = np.mean(cl[:, :2], axis=0).astype(np.float32)
                d = float(np.linalg.norm(cxy - xy.reshape(2)))
                if d < best_d:
                    best_d, best_z = d, _mean_z(cl)
            if best_z is not None:
                return best_z
            lst = self._object_map.detection_cloud.get(cls_name, [])
            best_d, best_z = 1e9, None
            for cl in lst:
                if not isinstance(cl, np.ndarray) or cl.shape[0] == 0:
                    continue
                cxy = np.mean(cl[:, :2], axis=0).astype(np.float32)
                d = float(np.linalg.norm(cxy - xy.reshape(2)))
                if d < best_d:
                    best_d, best_z = d, _mean_z(cl)
            return best_z

        for vp in cands:
            ok_all = True
            for rel in relations:
                if (not self._same_room_by_wall(vp, assign[rel["ref"]])) or (not self._same_room_by_wall(vp, assign[rel["tgt"]])):
                    ok_all = False
                    break
                rz = _z_at(rel["ref"], assign[rel["ref"]])
                tz = _z_at(rel["tgt"], assign[rel["tgt"]])
                if not self._relation_holds_at_pose(rel["rtype"], assign[rel["ref"]], assign[rel["tgt"]], vp, ref_z=rz, tgt_z=tz):
                    ok_all = False
                    break
            if ok_all:
                return True, vp
        return False, None

    def _verify_relations_on_stop_joint(
        self,
        radius_m: float = 3.0,
        require_joint_view: bool = True,
        include_context_only: bool = True,
        return_stats: bool = False,
    ):
        """
        At STOP, verify text relations jointly.
        Returns: "pass" | "fail" | "unknown" (or dict with stats if return_stats=True)
        """
        if not getattr(self, "_context_relations", None):
            return {"state": "unknown", "neighbor_count": 0, "relation_count": 0, "satisfied_count": 0} if return_stats else "unknown"

        approx = (self._object_map.last_target_coord if self._object_map.last_target_coord is not None
                  else (self._last_goal if np.linalg.norm(self._last_goal) > 0 else self._observations_cache.get("robot_xy", np.zeros(2, np.float32))))
        tgt_center = self._object_map.nearest_candidate_center(self._target_object, approx)

        k = self._cfg_rel_verify_max_inst_per_class
        neighbors = self._neighbor_context_instances(tgt_center, radius_m=radius_m, max_per_class=max(1, k))

        neighbor_cnt = sum(len(v) for v in neighbors.values())
        if len(self._context_categories) > 0 and neighbor_cnt == 0:
            return {"state": "fail", "neighbor_count": 0, "relation_count": 0, "satisfied_count": 0} if return_stats else "fail"

        rels = self._effective_relations_around_target(tgt_center, neighbors, include_context_only=include_context_only)
        if len(rels) == 0:
            return {"state": "unknown", "neighbor_count": neighbor_cnt, "relation_count": 0, "satisfied_count": 0} if return_stats else "unknown"

        # Build candidate assignment: target is fixed, context classes from neighbors
        cand: dict = {}
        cand[self._target_object] = [tgt_center]
        involved = set()
        for r in rels:
            involved.add(r["ref"])
            involved.add(r["tgt"])
        involved.discard(self._target_object)

        for cls in sorted(involved):
            pts = neighbors.get(cls, [])
            if len(pts) == 0:
                return "unknown"
            cand[cls] = pts

        # DFS backtracking for instance combination + joint viewpoint
        classes = [c for c in cand.keys() if c != self._target_object]
        assign = {self._target_object: tgt_center.copy()}

        def partial_prune_ok(assign_map: dict) -> bool:
            for r in rels:
                a = r["ref"]
                b = r["tgt"]
                if a in assign_map and b in assign_map:
                    pa, pb = assign_map[a], assign_map[b]
                    if not self._same_room_by_wall(pa, pb):
                        return False
                    if r["rtype"] == "near":
                        if float(np.linalg.norm(pa - pb)) > float(getattr(self, "_relation_max_distance_m", 2.0)):
                            return False
            return True

        def dfs(idx: int):
            if idx >= len(classes):
                ok, vp = self._find_joint_viewpoint_for_relations(rels, assign)
                return (ok, assign.copy() if ok else None, vp)
            cls = classes[idx]
            for p in cand[cls]:
                assign[cls] = p
                if not partial_prune_ok(assign):
                    assign.pop(cls, None)
                    continue
                ok, sol, vp = dfs(idx + 1)
                if ok:
                    return True, sol, vp
                assign.pop(cls, None)
            return False, None, None

        ok, sol, vp = dfs(0)
        if ok:
            return {"state": "pass", "neighbor_count": neighbor_cnt, "relation_count": len(rels), "satisfied_count": len(rels)} if return_stats else "pass"
        return {"state": "fail", "neighbor_count": neighbor_cnt, "relation_count": len(rels), "satisfied_count": 0} if return_stats else "fail"

    def _record_relctx_distance_for_promoted(self, prom_index: int, center_hint=None) -> None:
        """Compute and record relation-context distance sum for a promoted instance."""
        try:
            cx = self._object_map.promoted_center(self._target_object, int(prom_index))
            if cx is None:
                cx = np.asarray(center_hint, np.float32) if center_hint is not None else None
            if cx is None:
                return
            s = self._relctx_distance_sum(np.asarray(cx, np.float32).reshape(2))
            if np.isfinite(s):
                self._object_map.record_promoted_relctx_distance_sum(self._target_object, int(prom_index), float(s))
        except Exception:
            pass
