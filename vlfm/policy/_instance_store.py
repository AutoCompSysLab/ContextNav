# Instance state store mixin for BaseObjectNavPolicy

from typing import Any, Dict


class InstanceStoreMixin:
    """Per-instance state management for detection/promotion lifecycle."""

    def _inst__ensure(self, key: str) -> Dict[str, Any]:
        st = self._inst_state_store.get(key)
        if st is None:
            st = {
                "ynu_defer_active": False,
                "ynu_defer_start_step": 0,
                "ynu_buffer": [],
                "attr_defer_active": False,
                "attr_defer_start_step": 0,
                "attr_buffer": [],
                "ynu_unknown_counts": {},
                "ynu_passed_cats": set(),
                "pending_attr_categories": set(),
                "attr_decisions": {},
                "passed_exist": False,
                "passed_attrs": False,
            }
            self._inst_state_store[key] = st
        return st

    def _inst_pull(self, key: str) -> None:
        """Load per-instance state into shared fields for processing."""
        st = self._inst__ensure(key)
        self._active_inst_key = key
        self._ynu_defer_active = bool(st["ynu_defer_active"])
        self._ynu_defer_start_step = int(st["ynu_defer_start_step"])
        self._ynu_buffer = list(st["ynu_buffer"])
        self._attr_defer_active = bool(st["attr_defer_active"])
        self._attr_defer_start_step = int(st["attr_defer_start_step"])
        self._attr_buffer = list(st["attr_buffer"])
        self._ynu_unknown_counts = dict(st["ynu_unknown_counts"])
        self._ynu_passed_cats = set(st["ynu_passed_cats"])
        self._pending_attr_categories = set(st["pending_attr_categories"])

    def _inst_push(self, key: str) -> None:
        """Save shared fields back into per-instance state."""
        st = self._inst__ensure(key)
        st["ynu_defer_active"] = bool(self._ynu_defer_active)
        st["ynu_defer_start_step"] = int(self._ynu_defer_start_step)
        st["ynu_buffer"] = list(self._ynu_buffer)
        st["attr_defer_active"] = bool(self._attr_defer_active)
        st["attr_defer_start_step"] = int(self._attr_defer_start_step)
        st["attr_buffer"] = list(self._attr_buffer)
        st["ynu_unknown_counts"] = dict(self._ynu_unknown_counts)
        st["ynu_passed_cats"] = set(self._ynu_passed_cats)
        st["pending_attr_categories"] = set(self._pending_attr_categories)

    def _inst__new_temp_key(self, cls: str, det_idx: int) -> str:
        key = f"T:{cls}:{self._inst_key_next}"
        self._inst_key_next += 1
        self._det2store_key[(cls, int(det_idx))] = key
        return key

    def _inst__store_key_for(self, cls: str, det_idx: int, prom_idx: int) -> str:
        if int(prom_idx) >= 0:
            k = (cls, int(prom_idx))
            key = self._prom2store_key.get(k, None)
            if key is None:
                tmp = self._det2store_key.pop((cls, int(det_idx)), None)
                key = tmp if tmp is not None else f"P:{cls}:{int(prom_idx)}"
                self._prom2store_key[k] = key
            return key
        return self._det2store_key.get((cls, int(det_idx))) or self._inst__new_temp_key(cls, det_idx)

    def _inst_bind_after_commit(self, cls: str, det_idx: int, store_key: str, final_prom_idx: int) -> None:
        """After commit, link temp key (T) to promoted key (P). State stays at store_key."""
        if store_key is None or int(final_prom_idx) < 0:
            return
        self._prom2store_key[(cls, int(final_prom_idx))] = store_key
        self._det2store_key.pop((cls, int(det_idx)), None)
