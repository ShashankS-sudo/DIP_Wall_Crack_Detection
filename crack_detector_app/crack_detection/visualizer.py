import cv2
import numpy as np
from typing import Dict, Optional
from .geometry import CrackMetrics, WidthMeasurement


class CrackVisualizer:
    """
    Renders visual analysis, overlays, and diagnostic pipeline comparisons
    for crack inspection reporting.
    """

    def __init__(self):
        # Color definitions (BGR)
        self.COLOR_CRACK_MASK = (0, 0, 255)       # Red translucent overlay for crack body
        self.COLOR_SKELETON = (0, 255, 0)         # Bright Green for single-pixel skeleton
        self.COLOR_NORMAL_LINE = (255, 255, 0)    # Cyan for normal width measurement lines
        self.COLOR_MESH_BBOX = (255, 140, 0)      # Deep Sky Blue for network bounding box
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_BG_BOX = (20, 20, 20)

    def draw_measurement_overlay(
        self,
        original_bgr: np.ndarray,
        crack_mask: np.ndarray,
        skeleton_mask: np.ndarray,
        metrics: CrackMetrics,
        draw_normals: bool = True,
        draw_bboxes: bool = True,
    ) -> np.ndarray:
        """
        Creates a high-contrast visual inspection overlay on the original image:
        - Translucent colored crack segmentation mask
        - Single-pixel skeleton centerline
        - Orthogonal normal width measurement lines (Eq 19)
        - Mesh crack bounding box (Eq 22)
        - Top-left HUD summary banner
        """
        overlay = original_bgr.copy()
        h, w = overlay.shape[:2]

        # 1. Crack mask overlay (alpha blending)
        colored_mask = np.zeros_like(original_bgr)
        # Color code mask by severity
        if metrics.max_width_dmax_mm >= 0.5:
            mask_color = (0, 30, 230)    # Severe: Red
        elif metrics.max_width_dmax_mm >= 0.3:
            mask_color = (0, 140, 255)   # Moderate: Orange
        elif metrics.max_width_dmax_mm >= 0.1:
            mask_color = (0, 220, 255)   # Minor: Yellow
        else:
            mask_color = (0, 255, 120)   # Hairline: Mint Green

        colored_mask[crack_mask > 0] = mask_color
        alpha = 0.45
        cv2.addWeighted(colored_mask, alpha, overlay, 1.0 - alpha, 0, overlay)

        # 2. Skeleton centerline
        overlay[skeleton_mask > 0] = self.COLOR_SKELETON

        # 3. Normal measurement lines
        if draw_normals:
            for comp in metrics.components:
                for m in comp.measurements:
                    pt1 = (int(round(m.pt_a[0])), int(round(m.pt_a[1])))
                    pt2 = (int(round(m.pt_b[0])), int(round(m.pt_b[1])))
                    # Draw perpendicular measurement line across crack
                    cv2.line(overlay, pt1, pt2, self.COLOR_NORMAL_LINE, 1, cv2.LINE_AA)
                    cv2.circle(overlay, pt1, 1, (0, 255, 255), -1)
                    cv2.circle(overlay, pt2, 1, (0, 255, 255), -1)

        # 4. Network / Mesh crack bounding box (Eq 22)
        if draw_bboxes and metrics.total_unrefined_pixels_E > 0:
            for comp in metrics.components:
                if comp.is_network_crack or comp.num_unrefined_pixels_E > 300:
                    cv2.rectangle(
                        overlay,
                        (comp.bbox_x_min, comp.bbox_y_min),
                        (comp.bbox_x_max, comp.bbox_y_max),
                        self.COLOR_MESH_BBOX,
                        2,
                    )
                    tag = f"Mesh S={comp.mesh_area_S_mm2:.1f}mm2"
                    cv2.putText(
                        overlay,
                        tag,
                        (comp.bbox_x_min, max(15, comp.bbox_y_min - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        self.COLOR_MESH_BBOX,
                        1,
                        cv2.LINE_AA,
                    )

        # 5. Inset Info HUD Box
        hud_w, hud_h = min(360, w - 20), 125
        sub_img = overlay[10 : 10 + hud_h, 10 : 10 + hud_w]
        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        cv2.addWeighted(sub_img, 0.25, black_rect, 0.75, 0, sub_img)

        # Header
        cv2.putText(
            overlay,
            f"DIP Crack Inspection (mu={metrics.mu}mm/px)",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        # Stats
        cv2.putText(
            overlay,
            f"Length L: {metrics.total_skeleton_length_L_mm:.1f} mm | Area Q: {metrics.total_crack_area_Q_mm2:.1f} mm2",
            (18, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"Width: Max={metrics.max_width_dmax_mm:.2f}mm, Avg={metrics.avg_width_davg_mm:.2f}mm, Min={metrics.min_width_dmin_mm:.2f}mm",
            (18, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"Crack Rate CK: {metrics.crack_rate_CK_percent:.3f}% | PCI Index: {metrics.pavement_condition_index_PCI:.1f} ({metrics.damage_level})",
            (18, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"Severity: {metrics.severity_category}",
            (18, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            mask_color,
            1,
            cv2.LINE_AA,
        )

        return overlay

    def create_pipeline_comparison(
        self,
        original_bgr: np.ndarray,
        preprocessed_gray: np.ndarray,
        multiscale_map: np.ndarray,
        binary_mask: np.ndarray,
        skeleton_mask: np.ndarray,
        overlay_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Assembles a 2x3 publication-ready diagnostic grid showing every stage of
        the Digital Image Processing pipeline:
        (1) Original Image
        (2) Preprocessed / Illumination Equalized
        (3) Multiscale Feature Response Map
        (4) Segmented Binary Mask
        (5) Single-Pixel Skeleton Centerline
        (6) Quantitative Measurement & Severity Overlay
        """
        target_h, target_w = 320, 420

        def prep_panel(img: np.ndarray, title: str) -> np.ndarray:
            if len(img.shape) == 2:
                # Convert grayscale to 3-channel
                panel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                panel = img.copy()

            panel = cv2.resize(panel, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # Draw title banner on panel top
            cv2.rectangle(panel, (0, 0), (target_w, 28), (25, 25, 25), -1)
            cv2.putText(
                panel,
                title,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 200),
                1,
                cv2.LINE_AA,
            )
            return panel

        p1 = prep_panel(original_bgr, "1. Original Surface")
        p2 = prep_panel(preprocessed_gray, "2. CLAHE & Denoised")
        p3 = prep_panel(multiscale_map, "3. Multiscale Hessian Ridge")
        p4 = prep_panel(binary_mask, "4. Binary Crack Mask")
        p5 = prep_panel(skeleton_mask, "5. Refined Skeleton (Eq 16-18)")
        p6 = prep_panel(overlay_bgr, "6. Inspection & Width Overlay")

        row1 = np.hstack([p1, p2, p3])
        row2 = np.hstack([p4, p5, p6])
        grid = np.vstack([row1, row2])

        return grid
