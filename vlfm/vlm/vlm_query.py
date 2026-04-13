# vlfm/query/vlm_query.py
from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Union
import json
import re
import base64

import cv2
import numpy as np
from openai import OpenAI

class VLMQuery:
    """Vision-language model (VLM) query module for image+text scoring and Y/N/U decisions."""

    def __init__(self, base_url: str, vlm_model: str, ynu_T_low: int = 4, ynu_T_high: int = 11):
        self.client = OpenAI(api_key="ollama", base_url=base_url)
        self.vlm_model = vlm_model
        self.ynu_T_low = int(ynu_T_low)
        self.ynu_T_high = int(ynu_T_high)

    # ---------- Common utilities ----------
    def make_masked_image_url(self, rgb: np.ndarray, mask: np.ndarray, pad_ratio: float = 0.1) -> str:
        """Draw ROI contours on the image and return as a data URL."""
        try:
            binmask = (mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(binmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 3)
        except Exception:
            annotated_rgb = rgb
        _, buf = cv2.imencode(".jpg", annotated_rgb)
        b64 = base64.b64encode(buf).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def score_to_ynu(self, score: Optional[int]) -> str:
        if not isinstance(score, int):
            return "unknown"
        if score >= self.ynu_T_high:
            return "yes"
        if score <= self.ynu_T_low:
            return "no"
        return "unknown"

    # ---------- VLM calls ----------
    def vlm_prob_via_tools(self, prompt: str, image_url: str) -> Optional[float]:
        """Return probability [0,1] for image+text. Uses json_schema first, then forced JSON fallback."""
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_url},
            ],
        }]
        json_schema = {
            "name": "probability_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "probability": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["probability"],
                "additionalProperties": False,
            },
            "strict": True,
        }

        def _clamp01(x):
            try:
                return max(0.0, min(1.0, float(x)))
            except Exception:
                return None

        # Try schema first
        try:
            r = self.client.chat.completions.create(
                model=self.vlm_model,
                messages=msgs,
                response_format={"type": "json_schema", "json_schema": json_schema},
            )
            content = (r.choices[0].message.content or "").strip()
            data = json.loads(content)
            p = _clamp01(data.get("probability"))
            if p is not None:
                return p
        except Exception:
            pass

        # Free-form JSON fallback
        json_prompt = (
            prompt
            + '\n\nReturn ONLY a single JSON object exactly like: {"probability": <0..1>} with no extra text.'
        )
        try:
            r2 = self.client.chat.completions.create(
                model=self.vlm_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json_prompt},
                        {"type": "image_url", "image_url": image_url},
                    ],
                }],
            )
            txt = (r2.choices[0].message.content or "").strip()
            mjson = re.search(r"\{[\s\S]*\}", txt)
            if mjson:
                data = json.loads(mjson.group(0))
                p = _clamp01(data.get("probability"))
                if p is not None:
                    return p
            mnum = re.search(r"([01](?:\.\d+)?)", txt)
            if mnum:
                p = _clamp01(mnum.group(1))
                if p is not None:
                    return p
        except Exception:
            pass
        return None

    def score_0_15(self, prompt: str, image_url: str) -> Optional[int]:
        scoring_policy = (
            "You are a strict visual judge. Score whether the claim about the object is true "
            "using ONLY the provided image.\n"
            "SCORING (integer 0..15):\n"
            "- NO [0..4]\n- YES [11..15]\n- UNKNOWN [5..10] (visibility limits only). "
            "Never output text other than the JSON below."
        )
        msgs = [
            {"role": "system", "content": scoring_policy},
            {"role": "user", "content": [
                {"type": "text", "text": f'CLAIM: {prompt}\nReturn ONLY valid JSON: {{"score": <integer 0..15>}}'},
                {"type": "image_url", "image_url": image_url},
            ]},
        ]
        schema = {
            "name": "score_schema_0_15",
            "schema": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer", "minimum": 0, "maximum": 15},
                },
                "required": ["score"],
                "additionalProperties": False,
            },
            "strict": True,
        }
        try:
            r = self.client.chat.completions.create(
                model=self.vlm_model,
                messages=msgs,
                response_format={"type": "json_schema", "json_schema": schema},
                temperature=0.0,
            )
            data = json.loads((r.choices[0].message.content or "").strip())
            s = int(data.get("score"))
            if 0 <= s <= 15:
                return s
        except Exception:
            pass
        try:
            r2 = self.client.chat.completions.create(
                model=self.vlm_model, messages=msgs, temperature=0.0,
            )
            txt = (r2.choices[0].message.content or "").strip()
            m = re.search(r"\b(1[0-5]|[0-9])\b", txt)
            if m:
                s = int(m.group(1))
                if 0 <= s <= 15:
                    return s
        except Exception:
            pass
        return None

    def decision_ynu(self, prompt: str, image_url: str) -> str:
        s = self.score_0_15(prompt + " (Return a 0..15 integer confidence as JSON.)", image_url)
        return self.score_to_ynu(s)

    def attribute_scores_and_decisions(
        self,
        image_url: str,
        attr_questions: List[Dict[str, str]],
        categories: Optional[Set[str]] = None,
        ynu_passed_cats: Optional[Set[str]] = None,
    ) -> Tuple[Dict[str, str], Dict[str, int]]:
        """Per-attribute-category MAX score mapped to Y/N/U."""
        qs = [q for q in (attr_questions or []) if q.get("attribute") in ("color", "material", "shape")]
        if categories is not None:
            qs = [q for q in qs if q.get("attribute") in categories]

        grouped = {"color": [], "material": [], "shape": []}
        for it in qs:
            grouped[it["attribute"]].append(it["question"])

        decisions = {}
        scores = {}
        ynu_passed_cats = ynu_passed_cats or set()

        for cat, qlist in grouped.items():
            if cat in ynu_passed_cats:
                decisions[cat] = "yes"
                scores[cat] = self.ynu_T_high
                continue
            if not qlist:
                continue
            smax = None
            for q in qlist:
                s = self.score_0_15(f"Question about the outlined object: {q}", image_url)
                if isinstance(s, int):
                    smax = s if (smax is None or s > smax) else smax
            if smax is None:
                decisions[cat] = "unknown"
                continue
            scores[cat] = int(smax)
            decisions[cat] = self.score_to_ynu(int(smax))

        return decisions, scores

    def verify_non_coco_instances_exists(
        self,
        rgb: np.ndarray,
        inst_keys: List[Tuple[str, int]],
        threshold: float = 0.6,
        masks_by_instance: Optional[Dict[Tuple[str, int], np.ndarray]] = None
    ) -> Dict[Tuple[str, int], bool]:
        """
        Instance-level existence check with ROI-focused prompt.
        inst_keys: [(class_name, instance_idx), ...]
        masks_by_instance: {(class_name, instance_idx): mask}
        Returns: {(class_name, instance_idx): bool}
        """
        _, buf = cv2.imencode(".jpg", rgb)
        b64_full = base64.b64encode(buf).decode("utf-8")
        out: Dict[Tuple[str, int], bool] = {}

        def _q(cat: str) -> str:
            cat_q = cat.replace('"', '\\"')
            return (
                f'Is the outlined object an instance of the category "{cat_q}"?\n'
                "Judge ONLY the pixels inside the outline/mask; ignore everything else.\n"
                "If the outline is faint, assume the highlighted/outlined region marks a SINGLE candidate instance.\n"
                'Return ONLY JSON: {"prob": <float between 0 and 1>} where prob is P(instance \u2208 category).\n'
                "If the region is too small/blurred/occluded or ambiguous, return a value near 0.5."
            )

        for (c, i) in inst_keys:
            mask = None if masks_by_instance is None else masks_by_instance.get((c, i), None)
            if mask is None or np.count_nonzero(mask) == 0:
                url = f"data:image/jpeg;base64,{b64_full}"
            else:
                url = self.make_masked_image_url(rgb, (mask > 0).astype(np.uint8))

            prompt = _q(c)
            p = self.vlm_prob_via_tools(prompt, url) or 0.0
            try:
                p = float(max(0.0, min(1.0, p)))
            except Exception:
                p = 0.0

            out[(c, i)] = (p >= float(threshold))

        return out

