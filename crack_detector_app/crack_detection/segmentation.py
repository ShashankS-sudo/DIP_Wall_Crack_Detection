import cv2
import numpy as np
from typing import Dict, Tuple


class CrackSegmenter:
    """
    Segmentation and morphological noise filtering for building cracks.
    Purely classical digital image processing algorithms:
    - Adaptive & Otsu dual thresholding
    - Morphological bridge & closing operations
    - Highly optimized vectorized geometric property filtering
    """

    def __init__(
        self,
        threshold_mode: str = "hybrid",  # "hybrid", "otsu", "adaptive", "hysteresis"
        otsu_factor: float = 0.9,
        min_crack_area: int = 40,
        min_aspect_ratio: float = 1.5,
        closing_kernel_size: int = 3,
    ):
        self.threshold_mode = threshold_mode
        self.otsu_factor = otsu_factor
        self.min_crack_area = min_crack_area
        self.min_aspect_ratio = min_aspect_ratio
        self.closing_kernel_size = closing_kernel_size

    def binarize_multiscale(self, fused_map: np.ndarray) -> np.ndarray:
        """Converts the continuous multiscale crack response map into a binary crack mask."""
        if self.threshold_mode == "otsu":
            otsu_val, _ = cv2.threshold(fused_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = max(15, int(otsu_val * self.otsu_factor))
            _, binary = cv2.threshold(fused_map, thresh, 255, cv2.THRESH_BINARY)
            return binary

        elif self.threshold_mode == "adaptive":
            binary = cv2.adaptiveThreshold(
                fused_map,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=31,
                C=-5,
            )
            return binary

        elif self.threshold_mode == "hysteresis":
            otsu_val, _ = cv2.threshold(fused_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            high_thresh = min(245, max(30, int(otsu_val * 1.05)))
            low_thresh = max(12, int(otsu_val * 0.50))

            strong = fused_map >= high_thresh
            weak = (fused_map >= low_thresh) & (~strong)

            num_labels, labels = cv2.connectedComponents(strong.astype(np.uint8))
            dilated_labels = cv2.dilate(
                labels.astype(np.int32), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            )
            mask = strong | (weak & (dilated_labels > 0))
            return (mask * 255).astype(np.uint8)

        else:  # "hybrid" default: robust combination of Otsu and local adaptive thresholding
            otsu_val, _ = cv2.threshold(fused_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            effective_thresh = max(18, int(otsu_val * self.otsu_factor))
            _, binary_otsu = cv2.threshold(fused_map, effective_thresh, 255, cv2.THRESH_BINARY)

            binary_adaptive = cv2.adaptiveThreshold(
                fused_map,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=25,
                C=-4,
            )
            binary = cv2.bitwise_or(binary_otsu, binary_adaptive)
            return binary

    def morphological_refine(self, binary: np.ndarray) -> np.ndarray:
        """Applies mathematical morphology closing to bridge micro-fracture gaps."""
        k = self.closing_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return closed

    def filter_connected_components(self, binary: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        High-speed vectorized filtering of connected components:
        Filters out tiny dust/pores and keeps elongated or extensive crack structures.
        Runs in ~0.05s even on 9+ megapixel images.
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if num_labels <= 1:
            return np.zeros_like(binary), 0

        # Vectorized stats extraction
        areas = stats[1:, cv2.CC_STAT_AREA]
        widths = stats[1:, cv2.CC_STAT_WIDTH]
        heights = stats[1:, cv2.CC_STAT_HEIGHT]

        aspect_ratios = np.maximum(widths, heights) / (np.minimum(widths, heights) + 1e-5)

        # Keep components that satisfy crack geometry criteria:
        # 1. Elongated cracks (aspect ratio >= min_aspect_ratio) and area >= min_crack_area
        # 2. Or large crack networks (area >= 150)
        valid_indices = np.where(
            ((areas >= self.min_crack_area) & (aspect_ratios >= self.min_aspect_ratio))
            | (areas >= 160)
        )[0]

        valid_labels = valid_indices + 1
        count = len(valid_labels)

        # Fast LUT mapping
        lut = np.zeros(num_labels, dtype=np.uint8)
        lut[valid_labels] = 255
        clean_mask = lut[labels]

        return clean_mask, count

    def segment(self, fused_map: np.ndarray) -> Dict[str, np.ndarray]:
        """Runs the complete crack segmentation pipeline."""
        raw_binary = self.binarize_multiscale(fused_map)
        refined_binary = self.morphological_refine(raw_binary)
        clean_mask, count = self.filter_connected_components(refined_binary)

        return {
            "raw_binary": raw_binary,
            "refined_binary": refined_binary,
            "clean_mask": clean_mask,
            "component_count": count,
        }
