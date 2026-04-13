# vlfm/vlm/_llm_scoring.py
"""Scoring and ranking mixin for LLMQuery."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json
import re


class LLMScoringMixin:
    """Methods for detectability/signature scoring and IG ranking."""

    def _score_detectability_and_signature(
        self,
        categories: List[str],
        context_text: str,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Full 3-step fallback version: tools -> json_schema -> forced JSON parse."""
        if not categories:
            return {}, {}, {}

        client = self.client

        json_schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "detectability": {"type": "number", "minimum": 0, "maximum": 1},
                            "room_signature": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["category", "detectability", "room_signature"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["scores"],
            "additionalProperties": False,
        }

        system_prompt = (
            "You rate indoor object categories for navigation.\n"
            "- detectability: how large/salient and easy to see from normal viewpoints.\n"
            "- room_signature: how strongly the object indicates a specific room/zone "
            "(e.g., bed->bedroom, fridge->kitchen)."
        )
        user_base = (
            "Context text (object-centric relations may appear):\n"
            + str(context_text) + "\n"
            "Categories to score (snake_case): "
            + ", ".join(categories) + "\n"
        )

        # Step 1: tools (function-call)
        try:
            tools = [{
                "type": "function",
                "function": {
                    "name": "emit_scores",
                    "description": "Return scores for each category. All values must be in [0,1].",
                    "parameters": json_schema,
                },
            }]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_base + "Call the function with the JSON arguments."},
            ]

            r = client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "emit_scores"}},
            )

            msg = r.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                args_str = tool_calls[0].function.arguments or ""
                data = json.loads(args_str)
                return self._to_maps(data, categories)
        except Exception:
            pass

        # Step 2: response_format=json_schema
        try:
            schema_wrapper = {
                "name": "ig_schema",
                "schema": json_schema,
                "strict": True,
            }

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_base + "Return ONLY JSON using the provided schema."},
            ]

            r = client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                response_format={"type": "json_schema", "json_schema": schema_wrapper},
            )

            content = (r.choices[0].message.content or "").strip()
            data = json.loads(content)
            return self._to_maps(data, categories)
        except Exception:
            pass

        # Step 3: forced JSON parse (free-form -> JSON extraction)
        try:
            hard_user = (
                user_base
                + "Return ONLY a JSON object exactly matching this JSON Schema (no prose, no markdown):\n"
                + json.dumps(json_schema, ensure_ascii=False)
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": hard_user},
            ]
            r = client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.0,
            )
            raw = (r.choices[0].message.content or "").strip()
            json_str = self._extract_first_json_object(raw)
            if json_str:
                data = json.loads(json_str)
                return self._to_maps(data, categories)
        except Exception:
            pass

        return self._zero_maps(categories)

    def rank_ig_from_text(
        self,
        cats: List[str],
        rels: List[Dict[str, str]],
        vis_map: Dict[str, float],
        sig_map: Dict[str, float],
    ) -> List[Tuple[str, float]]:
        """
        Relation-type agnostic scoring:
          score(c) = w_freq * freq_norm(c) + w_vis * vis(c) + w_sig * sig(c)
        where freq_norm(c) = (#relations where c appears) / max_count.
        """
        freq = {c: 0 for c in cats}
        for it in (rels or []):
            a, b = it.get("ref", ""), it.get("tgt", "")
            if a in freq:
                freq[a] += 1
            if b in freq:
                freq[b] += 1
        max_cnt = max(freq.values()) if len(freq) > 0 else 0

        out: List[Tuple[str, float]] = []
        for c in cats:
            f = (freq[c] / max_cnt) if max_cnt > 0 else 0.0
            v = float(vis_map.get(c, 0.0))
            s = float(sig_map.get(c, 0.0))
            score = (
                float(getattr(self, "_ig_w_freq", 1.0)) * f
                + float(getattr(self, "_ig_w_vis", 0.7)) * v
                + float(getattr(self, "_ig_w_sig", 0.3)) * s
            )
            out.append((c, score))

        out.sort(key=lambda kv: kv[1], reverse=True)
        return out

    # ---- helper utilities ----

    @staticmethod
    def _clamp01(x: float) -> float:
        try:
            x = float(x)
            if x != x:  # NaN guard
                return 0.0
            return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
        except Exception:
            return 0.0

    def _to_maps(
        self, data: dict, cats: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        vis_map: Dict[str, float] = {}
        sig_map: Dict[str, float] = {}
        ig_map: Dict[str, float] = {}
        for it in data.get("scores", []):
            cat = str(it.get("category", "")).strip()
            if not cat:
                continue
            v = self._clamp01(it.get("detectability", 0.0))
            s = self._clamp01(it.get("room_signature", 0.0))
            vis_map[cat] = v
            sig_map[cat] = s
            ig_map[cat] = self.ig_w_vis * v + self.ig_w_sig * s
        for c in cats:
            vis_map.setdefault(c, 0.0)
            sig_map.setdefault(c, 0.0)
            ig_map.setdefault(c, 0.0)
        return vis_map, sig_map, ig_map

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[str]:
        """Extract the first top-level JSON object ({...}) from free text."""
        if not text:
            return None
        depth = 0
        start_idx = -1
        in_str = False
        esc = False
        for i, ch in enumerate(text):
            if ch == '"' and not esc:
                in_str = not in_str
            esc = (ch == '\\' and not esc) if in_str else False

            if not in_str:
                if ch == '{':
                    if depth == 0:
                        start_idx = i
                    depth += 1
                elif ch == '}':
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start_idx != -1:
                            return text[start_idx:i + 1]
        return None

    @staticmethod
    def _zero_maps(
        cats: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        vis = {c: 0.0 for c in cats}
        sig = {c: 0.0 for c in cats}
        ig = {c: 0.0 for c in cats}
        return vis, sig, ig
