# vlfm/vlm/llm_query.py
"""LLM query module -- main class inheriting scoring and extraction mixins."""
from __future__ import annotations
from typing import List
import re
import spacy

from openai import OpenAI

from vlfm.vlm._llm_scoring import LLMScoringMixin
from vlfm.vlm._llm_extraction import LLMExtractionMixin


class LLMQuery(LLMScoringMixin, LLMExtractionMixin):
    """
    Text LLM call module.
    Accepts an OpenAI-compatible client (e.g. Ollama gateway) and LLM model name.
    Scoring/ranking lives in _llm_scoring.py; extraction/grouping in _llm_extraction.py.
    """

    def __init__(
        self,
        base_url: str,
        llm_model: str,
        ig_w_vis: float = 0.7,
        ig_w_sig: float = 0.3,
    ) -> None:
        self.client = OpenAI(api_key="ollama", base_url=base_url)
        self.llm_model = llm_model
        self.ig_w_vis = float(ig_w_vis)
        self.ig_w_sig = float(ig_w_sig)
        self._spacy_nlp = None

    # ---------- spaCy / normalisation ----------
    def _get_spacy(self):
        if self._spacy_nlp is None:
            self._spacy_nlp = spacy.load("en_core_web_sm")
        return self._spacy_nlp

    def _norm_phrase(self, s: str) -> str:
        t = (s or "").lower().strip()
        t = t.replace("'s", "")
        t = re.sub(r"[_\-\u2013\u2014]+", " ", t)
        t = re.sub(r"^\s*(the|a|an)\s+", "", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    def _phrase_no_adj_num(self, chunk) -> str:
        """Strip adjectives, determiners, pronouns, particles, and numerals from a spaCy noun chunk."""
        kept = []
        for t in chunk:
            if t.pos_ in {"ADJ", "DET", "PRON", "PART", "NUM"}:
                continue
            if t.dep_ in {"amod", "det", "poss", "nummod"}:
                continue
            if t.pos_ in {"NOUN", "PROPN"} or t.dep_ == "compound":
                kept.append(t.lemma_.lower())
        return " ".join(kept).strip()

    def _extract_noun_phrases_via_spacy(self, text: str) -> List[str]:
        nlp = self._get_spacy()
        doc = nlp(text or "")
        out: List[str] = []
        seen: set = set()
        _STOP_NOUNS = {
            "left", "right", "front", "behind", "top", "bottom", "above", "below",
            "inside", "near", "between", "on",
            "area", "corner", "side", "part", "region", "direction",
            "item", "items", "thing", "things", "object", "objects", "stuff",
            "various", "another", "wall", "frame", "floor", "ceiling", "flooring",
            "room", "edge", "surrounding", "image", "pattern",
        }
        for np in doc.noun_chunks:
            base = self._phrase_no_adj_num(np)
            norm = self._norm_phrase(base)
            if not norm:
                continue
            if norm in _STOP_NOUNS:
                continue
            if norm not in seen:
                seen.add(norm)
                out.append(norm)
        return out
