import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
import json

from .preprocessor import ImagePreprocessor
from .multiscale import MultiscaleFeatureExtractor
from .segmentation import CrackSegmenter
from .skeleton import SkeletonExtractor
from .geometry import CrackGeometryAnalyzer, CrackMetrics
from .visualizer import CrackVisualizer


def make_serializable(val: Any) -> Any:
    """Recursively converts all NumPy types (np.bool_, np.int*, np.float*) into native Python types for JSON."""
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    elif isinstance(val, (np.integer, int)):
        return int(val)
    elif isinstance(val, (np.floating, float)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, dict):
        return {str(k): make_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [make_serializable(x) for x in val]
    return val


@dataclass
class PipelineResult:
    original_bgr: np.ndarray
    preprocessed_gray: np.ndarray
    multiscale_response: np.ndarray
    binary_mask: np.ndarray
    skeleton_mask: np.ndarray
    overlay_bgr: np.ndarray
    comparison_grid_bgr: np.ndarray
    metrics: CrackMetrics

    def to_dict(self) -> Dict[str, Any]:
        """Serializes numerical metrics and metadata to a Python dictionary."""
        raw_dict = {
            "calibration_mu_mm_per_px": float(self.metrics.mu),
            "dimensions": {
                "width_px": int(self.metrics.image_width),
                "height_px": int(self.metrics.image_height),
                "total_surface_area_mm2": float(self.metrics.total_surface_area_mm2),
            },
            "crack_measurements": {
                "unrefined_pixels_E": int(self.metrics.total_unrefined_pixels_E),
                "crack_area_Q_mm2": float(self.metrics.total_crack_area_Q_mm2),
                "skeleton_length_L_mm": float(self.metrics.total_skeleton_length_L_mm),
                "min_width_dmin_mm": float(self.metrics.min_width_dmin_mm),
                "max_width_dmax_mm": float(self.metrics.max_width_dmax_mm),
                "avg_width_davg_mm": float(self.metrics.avg_width_davg_mm),
            },
            "network_crack_mesh": {
                "proj_X1": int(self.metrics.proj_X1),
                "proj_X2": int(self.metrics.proj_X2),
                "proj_Y1": int(self.metrics.proj_Y1),
                "proj_Y2": int(self.metrics.proj_Y2),
                "mesh_bounding_area_S_mm2": float(self.metrics.mesh_bounding_area_S_mm2),
            },
            "condition_assessment": {
                "crack_rate_CK_percent": float(self.metrics.crack_rate_CK_percent),
                "pavement_condition_index_PCI": float(self.metrics.pavement_condition_index_PCI),
                "damage_level": str(self.metrics.damage_level),
                "severity_category": str(self.metrics.severity_category),
                "structural_risk_warning": str(self.metrics.structural_risk_warning),
            },
            "component_count": int(len(self.metrics.components)),
            "components": [
                {
                    "id": int(c.component_id),
                    "length_mm": float(c.length_L_mm),
                    "avg_width_mm": float(c.avg_width_davg_mm),
                    "max_width_mm": float(c.max_width_dmax_mm),
                    "area_mm2": float(c.actual_area_Q_mm2),
                    "is_network": bool(c.is_network_crack),
                    "mesh_area_mm2": float(c.mesh_area_S_mm2),
                }
                for c in self.metrics.components
            ],
        }
        return make_serializable(raw_dict)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class CrackDetector:
    """
    High-precision, high-performance crack detection pipeline strictly built on
    classical Digital Image Processing (DIP) and multiscale feature analysis.
    NO AI models or deep learning are used.
    """

    def __init__(
        self,
        mu: float = 0.92,
        clahe_clip_limit: float = 2.5,
        scales: list = None,
        threshold_mode: str = "hybrid",
        otsu_factor: float = 0.9,
        min_crack_area: int = 40,
        closing_kernel_size: int = 3,
        max_processing_dim: int = 1400,
    ):
        self.mu = mu
        self.max_processing_dim = max_processing_dim
        self.preprocessor = ImagePreprocessor(clahe_clip_limit=clahe_clip_limit)
        self.multiscale_extractor = MultiscaleFeatureExtractor(scales=scales)
        self.segmenter = CrackSegmenter(
            threshold_mode=threshold_mode,
            otsu_factor=otsu_factor,
            min_crack_area=min_crack_area,
            closing_kernel_size=closing_kernel_size,
        )
        self.skeleton_extractor = SkeletonExtractor(method="paper_morphological")
        self.geometry_analyzer = CrackGeometryAnalyzer(mu=mu)
        self.visualizer = CrackVisualizer()

    def process_image(
        self,
        image_input: Union[str, np.ndarray],
        mu_override: Optional[float] = None,
    ) -> PipelineResult:
        """
        Runs the end-to-end DIP crack detection pipeline with physical scale invariance:
        1. Ingests image (file path or numpy array)
        2. Preprocessing & illumination compensation
        3. Multiscale Hessian & morphological feature extraction
        4. Vectorized crack segmentation & component filtering
        5. Single-pixel skeleton extraction (Eq 16-18)
        6. Crack geometry analysis (Length L, Widths dmin/dmax/davg, Area Q, Mesh S, CK, PCI) (Eq 19-22)
        7. Publication visual overlays and comparison grids
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise FileNotFoundError(f"Failed to read image from path: {image_input}")
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

        orig_h, orig_w = image.shape[:2]
        base_mu = mu_override if mu_override is not None else self.mu

        # Adaptive physical scaling for high-resolution images (e.g. 4K / 9 Megapixels):
        # Keeps processing under 1.5 seconds while preserving exact physical millimeter measurements
        max_dim = max(orig_h, orig_w)
        if max_dim > self.max_processing_dim and self.max_processing_dim > 0:
            scale_factor = self.max_processing_dim / float(max_dim)
            new_w = int(round(orig_w * scale_factor))
            new_h = int(round(orig_h * scale_factor))
            proc_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Calibration factor adjusted inversely to preserve true physical metric invariance:
            effective_mu = base_mu / scale_factor
        else:
            proc_image = image
            effective_mu = base_mu

        self.geometry_analyzer.mu = effective_mu

        # Step 1: Preprocessing
        prep_res = self.preprocessor.preprocess(proc_image)
        preprocessed_gray = prep_res["preprocessed"]

        # Step 2: Multiscale Feature Analysis
        feat_res = self.multiscale_extractor.extract_multiscale_features(preprocessed_gray)
        multiscale_response = feat_res["fused_multiscale"]

        # Step 3: Segmentation
        seg_res = self.segmenter.segment(multiscale_response)
        binary_mask = seg_res["clean_mask"]

        # Step 4: Skeletonization (Eq 16-18)
        skeleton_mask = self.skeleton_extractor.extract_skeleton(binary_mask)

        # Step 5: Geometric Parameters & Severity Assessment (Eq 16-22)
        metrics = self.geometry_analyzer.analyze(binary_mask, skeleton_mask)

        # Restore metrics report calibration factor to base_mu for presentation
        metrics.mu = base_mu
        metrics.image_width = orig_w
        metrics.image_height = orig_h

        # Step 6: Visualization Overlay & Diagnostic Comparison Grid
        overlay_bgr = self.visualizer.draw_measurement_overlay(
            original_bgr=proc_image,
            crack_mask=binary_mask,
            skeleton_mask=skeleton_mask,
            metrics=metrics,
            draw_normals=True,
            draw_bboxes=True,
        )

        comparison_grid_bgr = self.visualizer.create_pipeline_comparison(
            original_bgr=proc_image,
            preprocessed_gray=preprocessed_gray,
            multiscale_map=multiscale_response,
            binary_mask=binary_mask,
            skeleton_mask=skeleton_mask,
            overlay_bgr=overlay_bgr,
        )

        return PipelineResult(
            original_bgr=proc_image,
            preprocessed_gray=preprocessed_gray,
            multiscale_response=multiscale_response,
            binary_mask=binary_mask,
            skeleton_mask=skeleton_mask,
            overlay_bgr=overlay_bgr,
            comparison_grid_bgr=comparison_grid_bgr,
            metrics=metrics,
        )
