"""
pipeline_core.py
================
Core abstractions for the modular OCR pipeline (Abstract Strategy Design Pattern).
Defines `PipelineContext`, `BasePipelineStage`, and `PipelineRunner`.
Includes per-stage and model-specific latency tracking for full execution visibility.
"""

import os
import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineCore")


@dataclass
class PipelineContext:
    """
    State container passed sequentially across all stages in the OCR pipeline.
    Preserves isolation between modules while building complete metadata and latency metrics.
    """
    image_path: str
    image_stem: str = ""
    original_image: Optional[np.ndarray] = field(default=None, repr=False)
    
    # Stage Latency Tracker
    stage_latencies: Dict[str, float] = field(default_factory=dict)
    total_pipeline_time_sec: float = 0.0
    
    # Stage 1: Detection outputs
    det_json_path: Optional[str] = None
    annotated_det_image_path: Optional[str] = None
    detections: List[Dict[str, Any]] = field(default_factory=list)
    det_inference_time_sec: float = 0.0
    
    # Stage 2: Crop outputs
    crops_dir: Optional[str] = None
    crop_manifest_path: Optional[str] = None
    crops_info: List[Dict[str, Any]] = field(default_factory=list)
    
    # Stage 3: Recognition outputs
    recog_full_json_path: Optional[str] = None
    recog_filtered_json_path: Optional[str] = None
    recog_results: List[Dict[str, Any]] = field(default_factory=list)
    filtered_recog_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Stage 4: CLIP Re-checking & Final outputs
    final_json_path: Optional[str] = None
    final_annotated_image_path: Optional[str] = None
    final_results: List[Dict[str, Any]] = field(default_factory=list)
    clip_inference_time_sec: float = 0.0
    
    # Stage 5: Regex Filter outputs
    regex_enriched_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Stage 6: NER outputs
    ner_enriched_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Stage 7: Classification outputs
    classified_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.image_stem and self.image_path:
            filename = os.path.basename(self.image_path)
            self.image_stem = os.path.splitext(filename)[0]


class BasePipelineStage(ABC):
    """
    Abstract Base Class for all pipeline processing stages.
    Enforces the Strategy Design Pattern for maximum modularity.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of the pipeline stage."""
        pass

    @abstractmethod
    def run(self, context: PipelineContext, config: Dict[str, Any]) -> PipelineContext:
        """
        Execute the stage logic on the given context using config settings.
        
        Args:
            context: PipelineContext object carrying data from previous stages.
            config: Central pipeline configuration dictionary.
            
        Returns:
            Updated PipelineContext object.
        """
        pass


class PipelineRunner:
    """
    Orchestrator that executes a sequence of BasePipelineStage instances on context objects,
    measuring precise latency per stage and overall pipeline processing time.
    """
    
    def __init__(self, stages: List[BasePipelineStage], config: Dict[str, Any]):
        self.stages = stages
        self.config = config
        
    def run_image(self, image_path: str) -> PipelineContext:
        """Run all configured stages on a single image file with precise per-stage timing."""
        logger.info(f"=== Starting Pipeline for image: {image_path} ===")
        context = PipelineContext(image_path=image_path)
        
        total_start = time.time()
        for stage in self.stages:
            logger.info(f"--> Running Stage: {stage.name}")
            stage_start = time.time()
            context = stage.run(context, self.config)
            stage_elapsed = time.time() - stage_start
            context.stage_latencies[stage.name] = round(stage_elapsed, 4)
            
        context.total_pipeline_time_sec = round(time.time() - total_start, 4)
        logger.info(f"=== Finished Pipeline for image: {image_path} in {context.total_pipeline_time_sec:.3f}s ===")
        return context

    def run_batch(self, image_paths: List[str]) -> List[PipelineContext]:
        """Run all configured stages on a batch of image files."""
        results = []
        for img_path in image_paths:
            ctx = self.run_image(img_path)
            results.append(ctx)
        return results
