"""
classification_stage.py
=======================
Semantic classification stage for the modular OCR pipeline.
Classifies each recognized text block as 'variable' (dynamic/personal content)
or 'static' (UI labels, buttons, fixed text) based on regex matches and NER entities.

Implements BasePipelineStage for seamless integration into the PipelineRunner.
"""

import os
import json
import time
from typing import List, Dict, Any, Set

from pipeline_core import BasePipelineStage, PipelineContext

# Default NER label sets (can be overridden via config)
DEFAULT_VARIABLE_NER_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "FAC", "NORP",
    "EVENT", "WORK_OF_ART", "PRODUCT", "LAW", "LANGUAGE"
}

DEFAULT_STATIC_NER_LABELS = {
    "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"
}


def classify_signal(item: dict, variable_ner_labels: Set[str] = None) -> str:
    """
    Classify a text block as 'variable' or 'static'.
    - 'variable': any text block that has regex matches OR NER entities recognized.
    - 'static': everything else (UI labels, buttons, unrecognized text).
    """
    if item.get("regex_matches"):
        return "variable"

    entities = item.get("entities", [])
    if entities:
        return "variable"

    return "static"


class ClassificationStage(BasePipelineStage):
    """
    Pipeline stage that classifies each text block as 'variable' or 'static'
    based on regex match and NER entity presence.

    Config keys (under 'classification'):
        enabled: bool (default True)
        variable_ner_labels: list of NER labels considered variable
        static_ner_labels: list of NER labels considered static
    """

    @property
    def name(self) -> str:
        return "ClassificationStage"

    def run(self, context: PipelineContext, config: Dict[str, Any]) -> PipelineContext:
        cls_cfg = config.get("classification", {})

        # Allow disabling the stage from config
        if not cls_cfg.get("enabled", True):
            print(f"[{self.name}] Skipped (disabled in config).")
            # Pass through results unchanged
            source = (
                context.ner_enriched_results
                or context.regex_enriched_results
                or context.final_results
                or context.filtered_recog_results
                or context.recog_results
            )
            context.classified_results = list(source)
            return context

        variable_labels = set(
            cls_cfg.get("variable_ner_labels", DEFAULT_VARIABLE_NER_LABELS)
        )

        # Pick the most recent results from the pipeline chain
        source = (
            context.ner_enriched_results
            or context.regex_enriched_results
            or context.final_results
            or context.filtered_recog_results
            or context.recog_results
        )

        if not source:
            print(f"[{self.name}] No results available for classification.")
            context.classified_results = []
            return context

        classified = []
        variable_count = 0
        static_count = 0

        for item in source:
            item_copy = dict(item)
            signal = classify_signal(item_copy, variable_labels)
            item_copy["final_signal"] = signal

            if signal == "variable":
                variable_count += 1
            else:
                static_count += 1

            classified.append(item_copy)

        context.classified_results = classified

        print(f"[{self.name}] Classified {len(classified)} blocks: "
              f"{variable_count} variable, {static_count} static.")
        return context
