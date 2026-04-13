# Target extraction mixin for BaseObjectNavPolicy
# Handles LLM-based attribute/context/relation extraction during _pre_step

from typing import List

from vlfm.vlm.coco_classes import COCO_CLASSES


class TargetExtractionMixin:
    """LLM attribute, context category, relation extraction, and detectability scoring."""

    def _extract_target_and_attributes(self, observations: dict) -> None:
        """
        Extract target object, attributes, context categories, relations, and IG ranking
        from the goal observations. Called once at episode reset.
        Handles both PSL (image filename) and COIN (objectgoal) benchmarks via
        CONTEXTNAV_BENCHMARK env var.
        """
        benchmark = self._cfg_benchmark

        self._target_object_attributes = observations['Goal Attributes']
        raw_goal = observations['Goal Attributes']

        # --- Target extraction differs by benchmark ---
        if benchmark == "psl":
            # PSL: extract from image filename
            self._target_object = observations['Goal Attributes']['image'].split('-')[-1].split('.')[0].strip()
            if self._target_object == 'tv_monitor':
                self._target_object = "tv"
            if self._target_object == 'plant':
                self._target_object = "potted plant"
            if self._target_object == 'sofa':
                self._target_object = "couch"
        else:
            # COIN and others: use objectgoal directly
            self._target_object = observations['objectgoal']

        # --- Attribute extraction (common) ---
        if isinstance(raw_goal, dict) and ("intrinsic_attributes" in raw_goal or "extrinsic_attributes" in raw_goal):
            self._target_description = str(raw_goal.get("intrinsic_attributes", "")) or ""
            self._context_description = str(raw_goal.get("extrinsic_attributes", "")) or ""
            self.t_full = self._target_description + self._context_description
            attrs_for_target = self._llm.extract_attributes(self._target_description, target=self._target_object)
        else:
            combined_text = str(raw_goal)
            try:
                attrs_for_target = self._llm.extract_attributes(combined_text, target=self._target_object)
            except Exception:
                attrs_for_target = {}
            self._target_description = self._llm.compose_target_sentence_from_attributes(self._target_object, attrs_for_target)
            self._context_description = combined_text

        # --- Build attribute questions ---
        if not attrs_for_target:
            self._attr_questions = []
        else:
            try:
                self._attr_questions = self._llm.build_attribute_questions(attrs_for_target, self._target_object)
            except Exception as e:
                print(f"[ATTR] question build error: {e}")
                self._attr_questions = []

        # --- Resolve context categories (synonyms removed, target excluded) ---
        self._resolve_context_categories()

        # --- Detectability / room-signature scoring ---
        self._score_detectability_and_signature()

        # --- Extract spatial relations ---
        self._extract_relations()

        # --- Build non-COCO caption ---
        self._build_non_coco_caption()

    def _resolve_context_categories(self) -> None:
        """Resolve unique context categories from description, excluding target."""
        try:
            _res = self._llm.resolve_context_categories(
                self._context_description,
                exclude_targets=self._target_object
            )
            self._context_categories = sorted(list(_res.get("category_set", set())))
            _mapping = _res.get("mapping", [])
            self._ctx_raw2canon = {raw: cat for (raw, cat, _meta) in _mapping}
            self._ctx_raw2canon[self._target_object] = self._target_object
            self._ctx_raw_terms = sorted(list({raw for (raw, _cat, _meta) in _mapping}))
        except Exception as e:
            print(f"[LLM] extract category failed: {e}")
            self._context_categories = []
            self._ctx_raw_terms = []
            self._ctx_raw2canon = {}

    def _score_detectability_and_signature(self) -> None:
        """Score context categories for detectability and room-signature via LLM."""
        try:
            self._vis_map, self._sig_map, self._ig_map = self._llm._score_detectability_and_signature(
                self._context_categories, self._context_description
            )
        except Exception:
            self._vis_map, self._sig_map, self._ig_map = {}, {}, {}

    def _extract_relations(self) -> None:
        """Extract spatial relations from description and compute IG ranking."""
        try:
            relation_obj_terms = self._context_categories + [self._target_object]
            rels = self._llm.extract_relations(self._context_description, relation_obj_terms, raw2canon=self._ctx_raw2canon)
        except Exception as e:
            print(f"[LLM] extract relations failed: {e}")
            rels = []
        self._context_relations = rels

        # Stage-0 text IG ranking
        ig_rank = self._llm.rank_ig_from_text(self._context_categories, rels, self._vis_map, self._sig_map)
        self._ig_high_categories = [k for k, _ in ig_rank[:max(0, 3)]]

    def _build_non_coco_caption(self) -> None:
        """Build GroundingDINO caption from non-COCO context categories."""
        try:
            noncoco_terms = [
                cat for cat in (self._context_categories or [])
                if cat not in COCO_CLASSES
            ]
            self._non_coco_caption = (" . ".join(noncoco_terms).strip() + " .")[:512]
            if self._target_object and self._target_object not in COCO_CLASSES and self._target_object not in noncoco_terms:
                noncoco_terms.insert(0, self._target_object)
            self._non_coco_caption_with_target = (" . ".join(noncoco_terms).strip() + " .")[:512]
        except Exception:
            self._non_coco_caption_with_target = ""
            self._non_coco_caption = ""
