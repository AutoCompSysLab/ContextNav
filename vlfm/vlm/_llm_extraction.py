# vlfm/vlm/_llm_extraction.py
"""Attribute extraction, object filtering, grouping, and relation extraction mixin."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import re

from vlfm.vlm.coco_classes import COCO_CLASSES


class LLMExtractionMixin:
    """Methods for extracting attributes, filtering objects, grouping, and relations."""

    # ------------------------------------------------------------------
    # extract_attributes
    # ------------------------------------------------------------------
    def extract_attributes(self, description: str, target: Optional[str] = None) -> Dict[str, str]:
        """Extract color/material/shape from a short object description."""
        sys = (
            "You extract visual attributes from a short object description. "
            "Find explicit attributes only if clearly stated. "
            "If a TARGET category is provided, STRICTLY return attributes that modify "
            "the TARGET and ignore attributes of any other object."
        )
        user_text = f"Text:\n{description}\n" + (f"TARGET CATEGORY: {target}\n" if target else "")
        tools = [{
            "type": "function",
            "function": {
                "name": "extract_attributes",
                "description": "Extract color/material/shape if present.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "color": {"type": "string"},
                        "material": {"type": "string"},
                        "shape": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        }]
        # 1) tools
        try:
            r = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": user_text}],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "extract_attributes"}},
            )
            msg = r.choices[0].message
            if getattr(msg, "tool_calls", None):
                args_json = msg.tool_calls[0].function.arguments
                data = json.loads(args_json) if isinstance(args_json, str) else args_json
                return {k: v.strip() for k, v in data.items() if isinstance(v, str) and v.strip()}
        except Exception:
            pass
        # 2) json_schema
        schema = {
            "name": "attr_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "color": {"type": "string"},
                    "material": {"type": "string"},
                    "shape": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "strict": True,
        }
        try:
            r = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": user_text}],
                response_format={"type": "json_schema", "json_schema": schema},
            )
            data = json.loads((r.choices[0].message.content or "").strip())
            return {k: v.strip() for k, v in data.items() if isinstance(v, str) and v.strip()}
        except Exception:
            pass
        # 3) forced
        forced = user_text + "\nReturn ONLY a JSON object with any keys color, material, shape (omit keys if not stated)."
        try:
            r = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": forced}],
            )
            txt = (r.choices[0].message.content or "").strip()
            m = re.search(r"\{[\s\S]*\}", txt)
            data = json.loads(m.group(0)) if m else {}
            return {k: v.strip() for k, v in data.items() if isinstance(v, str) and v.strip()}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # build_attribute_questions
    # ------------------------------------------------------------------
    def build_attribute_questions(self, attributes: Dict[str, str], category: str) -> List[Dict[str, str]]:
        """Convert attribute dict to yes/no VQA questions (only for present attributes)."""
        if not attributes:
            return []

        allowed = [a for a in ("color", "shape") if a in attributes]
        if not allowed:
            return []
        allowed_set = set(allowed)

        def _validate_questions(data: dict) -> List[Dict[str, str]]:
            """Filter parsed questions to only those with valid attributes."""
            out = []
            for it in data.get("questions", []):
                a = it.get("attribute")
                q = it.get("question")
                if a in allowed_set and isinstance(q, str) and q.strip():
                    out.append({"attribute": a, "question": q.strip()})
            return out

        sys = (
            "You convert attributes into direct yes/no VQA questions for a vision model. "
            "Write concise, unambiguous questions about a SINGLE instance of the target category. "
            "Only create questions for attributes that are present in the provided attributes JSON."
        )
        user = (
            "Target category: " + str(category) + "\n"
            "Attributes (JSON): " + json.dumps(attributes) + "\n"
            "Valid attributes you may ask about (must choose from these only): " + json.dumps(allowed) + "\n"
            'Return ONLY JSON: {"questions":[{"attribute":"<one of valid attributes>","question":"..."}]}'
        )

        # Dynamic enum restriction based on allowed attributes
        tools = [{
            "type": "function",
            "function": {
                "name": "return_questions",
                "description": "Return VQA yes/no questions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "attribute": {"type": "string", "enum": allowed},
                                    "question": {"type": "string"},
                                },
                                "required": ["attribute", "question"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["questions"],
                    "additionalProperties": False,
                },
            },
        }]

        try:
            r = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "return_questions"}},
            )
            msg = r.choices[0].message
            if getattr(msg, "tool_calls", None):
                data = json.loads(msg.tool_calls[0].function.arguments)
                return _validate_questions(data)
        except Exception:
            pass

        # json_schema fallback with enum restriction
        schema = {
            "name": "q_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "attribute": {"type": "string", "enum": allowed},
                                "question": {"type": "string"},
                            },
                            "required": ["attribute", "question"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["questions"],
                "additionalProperties": False,
            },
            "strict": True,
        }

        try:
            r2 = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": schema},
            )
            data = json.loads((r2.choices[0].message.content or "").strip())
            return _validate_questions(data)
        except Exception:
            pass

        # forced parse fallback
        try:
            r3 = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            )
            txt = (r3.choices[0].message.content or "").strip()
            m = re.search(r"\{[\s\S]*\}", txt)
            data = json.loads(m.group(0)) if m else {}
            return _validate_questions(data)
        except Exception:
            return []

    # ------------------------------------------------------------------
    # _filter_physical_objects
    # ------------------------------------------------------------------
    def _filter_physical_objects(self, noun_phrases: List[str]) -> List[str]:
        """
        Robust pipeline (temperature=0):
          1) tools (function-call) with enum constraint
          2) response_format=json_schema
          3) response_format=json_object
          4) forced JSON prompt (no code fences)
          5) heuristic filter (last resort)
        All steps return only a subset of the input list.
        """
        if not noun_phrases:
            return []

        # preserve order, deduplicate
        cand = list(dict.fromkeys(noun_phrases))
        cand_set = set(cand)

        # ---- local helpers ----
        def _strip_code_fences(text: str) -> str:
            text = re.sub(r"^```[\w-]*\s*", "", text.strip())
            text = re.sub(r"\s*```$", "", text)
            return text.strip()

        def _extract_json_block(text: str) -> str | None:
            text = _strip_code_fences(text)
            if text.strip().startswith("{") and text.strip().endswith("}"):
                return text.strip()
            for m in re.finditer(r"\{[\s\S]*?\}", text):
                block = m.group(0)
                if '"objects"' in block:
                    return block
            return None

        def _parse_objects_from_content(content: str) -> List[str]:
            if not content:
                return []
            try:
                data = json.loads(content)
            except Exception:
                blk = _extract_json_block(content) or ""
                try:
                    data = json.loads(blk)
                except Exception:
                    return []
            arr = data.get("objects", []) if isinstance(data, dict) else []
            return [s for s in arr if isinstance(s, str)]

        def _parse_objects_from_tool(resp) -> List[str]:
            try:
                msg = resp.choices[0].message
                tcs = getattr(msg, "tool_calls", None) or []
                if not tcs:
                    return []
                raw = tcs[0].function.arguments
                args = json.loads(raw) if isinstance(raw, str) else (raw or "{}")
                arr = args.get("objects", [])
                return [s for s in arr if isinstance(s, str)]
            except Exception:
                return []

        def _only_from_input_keep_order(items: List[str]) -> List[str]:
            seen = set()
            out = []
            for s in items:
                if s in cand_set and s not in seen:
                    seen.add(s)
                    out.append(s)
            return out

        # ---- shared system/user messages ----
        sys = (
            "Select which of the provided NOUN PHRASES refer to concrete PHYSICAL OBJECTS that can exist indoors. "
            "Exclude directions/regions/generic-only/material-only/color-only/shape-only entries. "
            "Return ONLY phrases from the input list-no new words."
        )
        user = "Candidates:\n" + json.dumps(cand, ensure_ascii=False)

        # 1) tools (function-call) with enum constraint
        tools = [{
            "type": "function",
            "function": {
                "name": "select_objects",
                "description": "Choose a subset of the provided candidates that are concrete indoor physical objects.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "objects": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": cand,
                            },
                        }
                    },
                    "required": ["objects"],
                    "additionalProperties": False,
                },
            },
        }]

        arr: List[str] = []
        try:
            r = self.client.chat.completions.create(
                model=self.llm_model,
                temperature=0,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": user}],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "select_objects"}},
            )
            arr = _parse_objects_from_tool(r)
        except Exception:
            arr = []

        # 2) response_format = json_schema
        if not arr:
            try:
                rf = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "select_objects_schema",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "objects": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": cand},
                                }
                            },
                            "required": ["objects"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                }
                r = self.client.chat.completions.create(
                    model=self.llm_model,
                    temperature=0,
                    messages=[{"role": "system", "content": sys},
                              {"role": "user", "content": user}],
                    response_format=rf,
                )
                arr = _parse_objects_from_content(r.choices[0].message.content or "")
            except Exception:
                arr = []

        # 3) response_format = json_object
        if not arr:
            try:
                r = self.client.chat.completions.create(
                    model=self.llm_model,
                    temperature=0,
                    messages=[{"role": "system", "content": sys},
                              {"role": "user", "content": user + '\nReturn only JSON of the form {"objects": [..]}.'}],
                    response_format={"type": "json_object"},
                )
                arr = _parse_objects_from_content(r.choices[0].message.content or "")
            except Exception:
                arr = []

        # 4) forced JSON prompt (no code fences)
        if not arr:
            try:
                force_sys = sys + " Answer in JSON only. No prose. No code fences."
                force_user = (
                    user + "\nReturn ONLY JSON exactly in this schema:\n"
                    '{"objects": [<subset of the given strings, keep original spelling>]}'
                )
                r = self.client.chat.completions.create(
                    model=self.llm_model,
                    temperature=0,
                    messages=[{"role": "system", "content": force_sys},
                              {"role": "user", "content": force_user}],
                )
                arr = _parse_objects_from_content(r.choices[0].message.content or "")
            except Exception:
                arr = []

        # 5) heuristic filter (last resort)
        if not arr:
            DIR = {"left", "right", "front", "behind", "top", "bottom", "above", "below",
                   "inside", "outside", "near", "between", "on", "off", "under", "over"}
            GENERIC = {"item", "items", "thing", "things", "object", "objects", "stuff",
                       "various", "another", "something", "anything", "everything",
                       "area", "side", "corner", "part", "region", "place"}
            MATERIALS = {"wood", "metal", "glass", "plastic", "fabric", "cloth", "leather",
                         "steel", "iron", "stone", "marble", "ceramic"}
            COLORS = {"red", "green", "blue", "black", "white", "gray", "grey", "yellow",
                      "brown", "purple", "orange", "pink", "beige", "cyan", "magenta"}
            SHAPES = {"round", "circular", "square", "rectangular", "rectangle", "triangle",
                      "triangular", "oval", "sphere", "spherical", "cylindrical"}

            out, seen = [], set()
            for s in cand:
                w = s.lower().strip()
                toks = w.split()
                # single token that is direction/generic/material/color/shape -> exclude
                if len(toks) == 1 and (w in DIR or w in GENERIC or w in MATERIALS or w in COLORS or w in SHAPES):
                    continue
                # all tokens are attribute words -> exclude
                if toks and all(t in MATERIALS or t in COLORS or t in SHAPES for t in toks):
                    continue
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            return out

        # filter model output: restrict to input set + preserve input order
        arr = _only_from_input_keep_order(arr)
        if hasattr(self, "_norm_phrase"):
            arr = list(dict.fromkeys([self._norm_phrase(s) for s in arr]))
            arr = _only_from_input_keep_order(arr)
        return arr

    # ------------------------------------------------------------------
    # _apply_coco_to_groups
    # ------------------------------------------------------------------
    def _apply_coco_to_groups(self, groups: Dict[str, List[str]], target: str) -> Dict[str, List[str]]:
        coco_set = set(COCO_CLASSES)
        t_can = self._norm_phrase(target or "")

        def _morph_base(s: str) -> str:
            s = self._norm_phrase(s)
            if s.endswith("ies"):
                return s[:-3] + "y"
            if s.endswith("ses") or s.endswith("xes") or s.endswith("zes") or s.endswith("ches") or s.endswith("shes"):
                return s[:-2]
            if s.endswith("s") and not s.endswith("ss"):
                return s[:-1]
            return s

        def _equiv(a, b) -> bool:
            return _morph_base(a) == _morph_base(b)

        merged: Dict[str, List[str]] = {}
        for canon, raws in groups.items():
            canon_norm = self._norm_phrase(canon)
            # keep target word as-is without morphological correction
            if _equiv(canon_norm, t_can):
                chosen = target
            else:
                chosen = None
                for t in [canon] + list(raws):
                    n = self._norm_phrase(t)
                    if n in coco_set:
                        chosen = n
                        break
                if chosen is None:
                    chosen = canon_norm
            merged.setdefault(chosen, [])
            for r in raws:
                nr = self._norm_phrase(r)
                if nr not in merged[chosen]:
                    merged[chosen].append(nr)
        return merged

    # ------------------------------------------------------------------
    # _unify_objects_via_llm_space
    # ------------------------------------------------------------------
    def _unify_objects_via_llm_space(self, objects: List[str], target: str) -> Dict[str, List[str]]:
        if not objects:
            return {}
        sys = (
            "Group INDOOR household object names by synonyms.\n"
            "- Use SPACE-BASED lowercased canonical labels (no underscores).\n"
            "IMPORTANT: Synonym lists MUST be drawn ONLY from the provided input terms.\n"
            f"- CRITICAL: Target category is '{self._norm_phrase(target)}'. "
            "If ANY group contains this target, set the canonical key EXACTLY to the target string as given."
        )
        user = (
            "Objects:\n" + json.dumps(objects, ensure_ascii=False) + "\n"
            + "COCO class names (space-based):\n" + json.dumps(COCO_CLASSES, ensure_ascii=False)
        )

        def _postproc(data_obj):
            """
            Accepts either:
              {"mapping": {"lamp": ["ceiling lamp"], ...}}
            or top-level object:
              {"lamp": [...], "counter": [...]}
            """
            mp = data_obj.get("mapping") if isinstance(data_obj, dict) else None
            if mp is None and isinstance(data_obj, dict):
                mp = data_obj
            if not isinstance(mp, dict):
                return {}
            cand = set(objects)
            out = {}
            for k, v in mp.items():
                canon = self._norm_phrase(k)
                syns = []
                seen = set()
                for s in (v or []):
                    n = self._norm_phrase(s)
                    if n in cand and n not in seen:
                        seen.add(n)
                        syns.append(n)
                if canon and syns:
                    out[canon] = syns
            return out

        # 1) tools (function-call)
        tools = [{
            "type": "function",
            "function": {
                "name": "return_groups",
                "description": "Return synonym groups for household objects (space-based canonical).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mapping": {
                            "type": "object",
                            "additionalProperties": {"type": "array", "items": {"type": "string"}},
                        }
                    },
                    "required": ["mapping"],
                    "additionalProperties": False,
                },
            },
        }]
        try:
            r = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "return_groups"}},
            )
            msg = r.choices[0].message
            if getattr(msg, "tool_calls", None):
                args_json = msg.tool_calls[0].function.arguments
                data = json.loads(args_json) if isinstance(args_json, str) else args_json
                mp = _postproc(data)
                if mp:
                    return mp
        except Exception:
            pass

        # 2) response_format=json_schema
        schema = {
            "name": "group_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "mapping": {
                        "type": "object",
                        "additionalProperties": {"type": "array", "items": {"type": "string"}},
                    }
                },
                "required": ["mapping"],
                "additionalProperties": False,
            },
            "strict": True,
        }
        try:
            r2 = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": schema},
            )
            data = json.loads((r2.choices[0].message.content or "").strip())
            mp = _postproc(data)
            if mp:
                return mp
        except Exception:
            pass

        # 3) forced JSON
        forced = user + (
            "\nReturn ONLY a single JSON OBJECT where each key is the space-based canonical label "
            "and each value is an array of input synonyms. Example:\n"
            '{\n  "kitchen island": ["kitchen island"],\n'
            '  "counter": ["counter", "countertop"],\n'
            '  "lamp": ["ceiling lamp"]\n}'
        )
        try:
            r3 = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": forced}],
            )
            txt = (r3.choices[0].message.content or "").strip()
            m = re.search(r"\{[\s\S]*\}", txt)
            data = json.loads(m.group(0)) if m else {}
            mp = _postproc(data)
            if mp:
                return mp
        except Exception:
            pass

        # fallback: identity mapping
        out: Dict[str, List[str]] = {}
        for s in objects:
            c = self._norm_phrase(s)
            out.setdefault(c, [])
            if s not in out[c]:
                out[c].append(s)
        return out

    # ------------------------------------------------------------------
    # resolve_context_categories
    # ------------------------------------------------------------------
    def resolve_context_categories(
        self,
        context_text: str,
        exclude_targets: Optional[str | List[str]] = None,
    ) -> Dict[str, Any]:
        noun_phrases = self._extract_noun_phrases_via_spacy(context_text or "")
        objects = self._filter_physical_objects(noun_phrases)
        anchor_terms: List[str] = []
        if exclude_targets:
            anchor_terms = (
                [exclude_targets] if isinstance(exclude_targets, str)
                else [t for t in exclude_targets if isinstance(t, str)]
            )
        objects_for_unify = list(dict.fromkeys(objects + anchor_terms))

        groups0 = self._unify_objects_via_llm_space(objects_for_unify, target=exclude_targets)
        groups = self._apply_coco_to_groups(groups0, target=exclude_targets)

        excl_norm: set = set()
        if exclude_targets:
            if isinstance(exclude_targets, str):
                excl_norm = {self._norm_phrase(exclude_targets)}
            else:
                excl_norm = {self._norm_phrase(t) for t in exclude_targets if isinstance(t, str)}

        def _skip_group(canon: str, raws: List[str]) -> bool:
            if not excl_norm:
                return False
            c = self._norm_phrase(canon)
            if c in excl_norm or any((c in e or e in c) for e in excl_norm):
                return True
            rn = [self._norm_phrase(r) for r in raws]
            if any(r in excl_norm for r in rn):
                return True
            return False

        mapping: list = []
        raw_set: set = set()
        kept_canons: set = set()
        for canon, raws in groups.items():
            if _skip_group(canon, raws):
                continue
            kept_canons.add(canon)
            for r in raws:
                rr = self._norm_phrase(r)
                raw_set.add(rr)
                mapping.append((rr, canon, {"source": "llm"}))

        return {"category_set": kept_canons, "mapping": mapping, "raw_terms": sorted(list(raw_set))}

    # ------------------------------------------------------------------
    # extract_relations
    # ------------------------------------------------------------------
    def extract_relations(
        self,
        context_text: str,
        allowed_raw: List[str],
        raw2canon: Dict[str, str],
    ) -> List[Dict[str, str]]:
        client = self.client

        def _norm_surface(text: str) -> str:
            t = text.lower().strip()
            t = re.sub(r"[-\u2013\u2014]+", " ", t)
            t = re.sub(r"\s+", " ", t)
            t = t.replace("'s", "")
            return t

        _R = {"front", "behind", "left", "right", "near", "below"}
        sys = (
            "You convert a single-view image caption into PAIRWISE spatial relations among INDOOR OBJECTS.\n"
            "The caption often uses FRAME-RELATIVE absolute positions (e.g., \"left side of the image\", "
            "\"upper right corner\", \"bottom\", \"top\").\n\n"
            "INTERPRETATION RULES (CRITICAL):\n"
            "1) left/right: If object A is described on the LEFT (or left/upper-left/lower-left) of the FRAME and "
            "object B on the RIGHT (or right/upper-right/lower-right), output {ref:A, tgt:B, rtype:left}.\n"
            "2) below: If A is in LOWER/bottom region and B in UPPER/top region of the FRAME, output {ref:A, tgt:B, rtype:below}.\n"
            "   (below means A appears LOWER in the image than B -- do NOT reason about depth; only 2D frame ordering.)\n"
            "3) If both left/right AND below are entailed for the same pair, output TWO relations (one per rtype).\n"
            "4) Use ONLY the provided Allowed raw terms as surface forms for 'ref' and 'tgt'. "
            "If the text uses a synonym or variant not in the list (e.g., 'sofa'), map it to the NEAREST allowed label "
            "(e.g., 'couch') and output the allowed label.\n"
            "5) Emit only relations that are CLEARLY ENTAILED. If ambiguous or under-specified, SKIP.\n"
            "6) Avoid contradictory pairs (e.g., left(A,B) and left(B,A)).\n"
            "7) Prefer concise sets: emit at most 6 relations, focusing on the most salient pairs.\n\n"
            "OUTPUT FORMAT: JSON only, schema below."
        )
        user = (
            f"Text:\n{context_text}\n\nAllowed raw terms (verbatim list): {json.dumps(allowed_raw)}\n"
            'Return ONLY JSON: {"relations":[{"ref":"<one of allowed raw terms>",'
            '"tgt":"<one of allowed raw terms>","rtype":"front|behind|left|right|near|below"}]}'
        )
        try:
            r = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            )
            txt = (r.choices[0].message.content or "").strip()
            m = re.search(r"\{[\s\S]*\}", txt)
            data = json.loads(m.group(0)) if m else {"relations": []}
        except Exception:
            data = {"relations": []}

        rels: List[Dict[str, str]] = []
        for it in data.get("relations", []):
            ref_raw = _norm_surface(str(it.get("ref", "")))
            tgt_raw = _norm_surface(str(it.get("tgt", "")))
            rtype = str(it.get("rtype", "")).strip().lower()
            if rtype not in _R:
                continue
            ref_can = raw2canon.get(ref_raw)
            tgt_can = raw2canon.get(tgt_raw)
            if ref_can and tgt_can:
                rels.append({"ref": ref_can, "tgt": tgt_can, "rtype": rtype})
        return rels

    # ------------------------------------------------------------------
    # compose_target_sentence_from_attributes
    # ------------------------------------------------------------------
    def compose_target_sentence_from_attributes(self, target: str, attrs: Dict[str, str]) -> str:
        """
        Compose an English sentence from target object attributes.
        E.g. "a red chair made of wood with a round shape."
        Returns "a <target>." if attributes are empty.
        """
        def _first_alias(t: str) -> str:
            return (t or "").split("|")[0].strip() or "object"

        def _article_for(phrase: str) -> str:
            w = (phrase or "").strip().lower()
            return "an" if w[:1] in {"a", "e", "i", "o", "u"} else "a"

        tgt = _first_alias(target)
        color = (attrs.get("color", "") or "").strip()
        material = (attrs.get("material", "") or "").strip()
        shape = (attrs.get("shape", "") or "").strip()

        base_noun = f"{color} {tgt}".strip() if color else tgt
        sent = f"{_article_for(base_noun)} {base_noun}"
        if material:
            sent += f" made of {material}"
        if shape:
            sent += (" with a " + shape if "shape" in shape.lower() else f" with a {shape} shape")
        return (sent + ".").strip()
