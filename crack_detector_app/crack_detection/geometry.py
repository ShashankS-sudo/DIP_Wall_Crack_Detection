import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


@dataclass
class WidthMeasurement:
    skeleton_pt: Tuple[int, int]
    pt_a: Tuple[float, float]
    pt_b: Tuple[float, float]
    width_pixels: float
    width_mm: float
    normal_angle_deg: float


@dataclass
class CrackComponentData:
    component_id: int
    num_unrefined_pixels_E: int
    actual_area_Q_mm2: float
    skeleton_pixel_length_l: float
    length_L_mm: float
    min_width_dmin_mm: float
    max_width_dmax_mm: float
    avg_width_davg_mm: float
    bbox_x_min: int
    bbox_x_max: int
    bbox_y_min: int
    bbox_y_max: int
    mesh_area_S_mm2: float
    is_network_crack: bool
    measurements: List[WidthMeasurement] = field(default_factory=list)


@dataclass
class CrackMetrics:
    # Calibration factor mu (mm per pixel, default 0.92 mm as in paper)
    mu: float
    image_width: int
    image_height: int
    total_surface_area_mm2: float

    # Global crack aggregates
    total_unrefined_pixels_E: int
    total_crack_area_Q_mm2: float
    total_skeleton_length_L_mm: float
    min_width_dmin_mm: float
    max_width_dmax_mm: float
    avg_width_davg_mm: float

    # Mesh / Network crack bounding parameters
    proj_X1: int
    proj_X2: int
    proj_Y1: int
    proj_Y2: int
    mesh_bounding_area_S_mm2: float

    # Indices
    crack_rate_CK_percent: float
    pavement_condition_index_PCI: float
    damage_level: str  # "Good", "Satisfactory", "Fair", "Poor", "Severe", "Critical"
    severity_category: str  # "Hairline (<0.1mm)", "Minor (0.1-0.3mm)", "Moderate (0.3-0.5mm)", "Severe (>0.5mm)"
    structural_risk_warning: str

    # Per-component details
    components: List[CrackComponentData] = field(default_factory=list)


class CrackGeometryAnalyzer:
    """
    Implements crack geometric parameter calculations strictly according to
    Section 3/4 (Equations 16-22) of the paper.
    Vectorized and optimized for high-resolution images.
    """

    def __init__(self, mu: float = 0.92, max_sample_points_per_comp: int = 60):
        self.mu = mu
        self.max_sample_points_per_comp = max_sample_points_per_comp

    def compute_skeleton_length(self, skeleton_mask: np.ndarray) -> Tuple[float, float]:
        """
        Fast vectorized skeleton pixel length l and physical length L = mu * l.
        Uses 2D convolution to account for 8-connectivity (diagonal = sqrt(2), orthogonal = 1.0).
        """
        num_skel = cv2.countNonZero(skeleton_mask)
        if num_skel == 0:
            return 0.0, 0.0

        skel_bin = (skeleton_mask > 0).astype(np.uint8)
        k_ortho = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        k_diag = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.uint8)

        ortho_neighbors = cv2.filter2D(skel_bin, -1, k_ortho, borderType=cv2.BORDER_CONSTANT)
        diag_neighbors = cv2.filter2D(skel_bin, -1, k_diag, borderType=cv2.BORDER_CONSTANT)

        # Count connections once per undirected edge
        mask = skel_bin > 0
        ortho_edges = np.sum(ortho_neighbors[mask]) / 2.0
        diag_edges = np.sum(diag_neighbors[mask]) / 2.0

        step_len = ortho_edges * 1.0 + diag_edges * np.sqrt(2.0)
        if step_len < num_skel * 0.5:
            step_len = float(num_skel)

        length_L_mm = self.mu * step_len
        return step_len, length_L_mm

    def compute_vertical_line_widths(
        self,
        skeleton_pts: np.ndarray,
        dist_transform: np.ndarray,
        crack_mask: np.ndarray,
    ) -> List[WidthMeasurement]:
        """
        Generates normal line measurements (Eq 19-20) for visual overlay and verification.
        Uses distance transform to find exact normal boundary intersections without slow raycasting.
        """
        measurements: List[WidthMeasurement] = []
        n_pts = len(skeleton_pts)
        if n_pts == 0:
            return measurements

        h, w = crack_mask.shape
        stride = max(1, n_pts // self.max_sample_points_per_comp)
        sample_indices = range(0, n_pts, stride)

        for idx in sample_indices:
            cy, cx = skeleton_pts[idx]
            local_r = float(dist_transform[cy, cx])
            if local_r <= 0.5:
                local_r = 0.5

            d_pixels = local_r * 2.0
            d_mm = d_pixels * self.mu

            # Local tangent estimation in 5x5 window
            win = 3
            y1, y2 = max(0, cy - win), min(h, cy + win + 1)
            x1, x2 = max(0, cx - win), min(w, cx + win + 1)
            nbrs = np.argwhere(crack_mask[y1:y2, x1:x2] > 0)

            if len(nbrs) >= 2:
                coords = nbrs.astype(np.float32)
                cov = np.cov(coords, rowvar=False)
                if cov.ndim == 2:
                    _, eigvecs = np.linalg.eigh(cov)
                    norm_y, norm_x = float(eigvecs[0, 0]), float(eigvecs[1, 0])
                else:
                    norm_y, norm_x = 1.0, 0.0
            else:
                norm_y, norm_x = 1.0, 0.0

            nmag = np.sqrt(norm_y ** 2 + norm_x ** 2) + 1e-6
            norm_y /= nmag
            norm_x /= nmag

            pt_a = (float(np.clip(cx + norm_x * local_r, 0, w - 1)),
                    float(np.clip(cy + norm_y * local_r, 0, h - 1)))
            pt_b = (float(np.clip(cx - norm_x * local_r, 0, w - 1)),
                    float(np.clip(cy - norm_y * local_r, 0, h - 1)))

            angle_deg = float(np.degrees(np.arctan2(norm_y, norm_x)))

            measurements.append(
                WidthMeasurement(
                    skeleton_pt=(int(cx), int(cy)),
                    pt_a=pt_a,
                    pt_b=pt_b,
                    width_pixels=float(d_pixels),
                    width_mm=float(d_mm),
                    normal_angle_deg=angle_deg,
                )
            )

        return measurements

    def analyze(
        self,
        crack_mask: np.ndarray,
        skeleton_mask: np.ndarray,
    ) -> CrackMetrics:
        """Runs fast quantitative DIP analysis following Equations 16-22."""
        h, w = crack_mask.shape
        total_surface_area_mm2 = (h * self.mu) * (w * self.mu)

        # Eq 21: Unrefined crack pixels E = sum sum f(x, y)
        total_E = int(cv2.countNonZero(crack_mask))
        # Actual crack area Q = mu^2 * E
        total_Q_mm2 = (self.mu ** 2) * float(total_E)

        if total_E == 0:
            return CrackMetrics(
                mu=self.mu,
                image_width=w,
                image_height=h,
                total_surface_area_mm2=round(total_surface_area_mm2, 2),
                total_unrefined_pixels_E=0,
                total_crack_area_Q_mm2=0.0,
                total_skeleton_length_L_mm=0.0,
                min_width_dmin_mm=0.0,
                max_width_dmax_mm=0.0,
                avg_width_davg_mm=0.0,
                proj_X1=0,
                proj_X2=0,
                proj_Y1=0,
                proj_Y2=0,
                mesh_bounding_area_S_mm2=0.0,
                crack_rate_CK_percent=0.0,
                pavement_condition_index_PCI=100.0,
                damage_level="Good",
                severity_category="None (No Cracks)",
                structural_risk_warning="Surface is structurally intact with no detectable cracks.",
                components=[],
            )

        # Fast Euclidean Distance Transform to compute local radius at each skeleton pixel
        dist_transform = cv2.distanceTransform(crack_mask, cv2.DIST_L2, 5)

        # Fast connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            crack_mask, connectivity=8
        )

        component_list: List[CrackComponentData] = []
        all_measurements: List[WidthMeasurement] = []
        all_skel_pts = np.argwhere(skeleton_mask > 0)

        # Global width statistics from all skeleton pixels directly via distance transform
        if len(all_skel_pts) > 0:
            all_radii = dist_transform[all_skel_pts[:, 0], all_skel_pts[:, 1]]
            all_widths_mm = all_radii * 2.0 * self.mu
            global_dmin = float(np.min(all_widths_mm))
            global_dmax = float(np.max(all_widths_mm))
        else:
            global_dmin = 0.0
            global_dmax = 0.0

        total_length_L_mm = 0.0
        min_x_global, max_x_global = w, 0
        min_y_global, max_y_global = h, 0

        # Process top significant components (up to 50)
        sorted_labels = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
        top_labels = sorted_labels[:50]

        for label in top_labels:
            comp_E = stats[label, cv2.CC_STAT_AREA]
            if comp_E < 20:
                continue

            bx = stats[label, cv2.CC_STAT_LEFT]
            by = stats[label, cv2.CC_STAT_TOP]
            bw = stats[label, cv2.CC_STAT_WIDTH]
            bh = stats[label, cv2.CC_STAT_HEIGHT]

            x1, x2 = bx, bx + bw
            y1, y2 = by, by + bh
            min_x_global = min(min_x_global, x1)
            max_x_global = max(max_x_global, x2)
            min_y_global = min(min_y_global, y1)
            max_y_global = max(max_y_global, y2)

            comp_Q = (self.mu ** 2) * float(comp_E)
            comp_mesh_S = (self.mu ** 2) * float(bw * bh)
            is_network = bool(bw > 60 and bh > 60 and (bw * bh) > (comp_E * 2.5))

            # Local skeleton slice
            skel_roi = skeleton_mask[by:y2, bx:x2]
            dist_roi = dist_transform[by:y2, bx:x2]
            crack_roi = crack_mask[by:y2, bx:x2]

            comp_l, comp_L_mm = self.compute_skeleton_length(skel_roi)
            total_length_L_mm += comp_L_mm

            local_skel_pts = np.argwhere(skel_roi > 0)
            if len(local_skel_pts) > 0:
                local_radii = dist_roi[local_skel_pts[:, 0], local_skel_pts[:, 1]]
                comp_dmin = float(np.min(local_radii) * 2.0 * self.mu)
                comp_dmax = float(np.max(local_radii) * 2.0 * self.mu)
            else:
                comp_dmin = comp_dmax = self.mu

            comp_davg = (comp_Q / comp_L_mm) if comp_L_mm > 0 else comp_dmax

            # Sample visual measurements
            # Adjust local coordinates to global
            if len(local_skel_pts) > 0:
                global_local_pts = local_skel_pts + np.array([by, bx])
                comp_measurements = self.compute_vertical_line_widths(
                    global_local_pts, dist_transform, crack_mask
                )
            else:
                comp_measurements = []

            all_measurements.extend(comp_measurements)

            component_list.append(
                CrackComponentData(
                    component_id=int(label),
                    num_unrefined_pixels_E=int(comp_E),
                    actual_area_Q_mm2=round(comp_Q, 2),
                    skeleton_pixel_length_l=round(comp_l, 1),
                    length_L_mm=round(comp_L_mm, 2),
                    min_width_dmin_mm=round(comp_dmin, 3),
                    max_width_dmax_mm=round(comp_dmax, 3),
                    avg_width_davg_mm=round(comp_davg, 3),
                    bbox_x_min=int(x1),
                    bbox_x_max=int(x2),
                    bbox_y_min=int(y1),
                    bbox_y_max=int(y2),
                    mesh_area_S_mm2=round(comp_mesh_S, 2),
                    is_network_crack=is_network,
                    measurements=comp_measurements,
                )
            )

        # Average width: d_avg = Q / L (Eq 21)
        if total_length_L_mm > 0:
            global_davg = total_Q_mm2 / total_length_L_mm
        else:
            global_davg = 0.0

        # Global mesh area (Eq 22)
        if max_x_global > min_x_global and max_y_global > min_y_global:
            mesh_area_S = (self.mu ** 2) * float((max_x_global - min_x_global) * (max_y_global - min_y_global))
            proj_x1, proj_x2 = min_x_global, max_x_global
            proj_y1, proj_y2 = min_y_global, max_y_global
        else:
            mesh_area_S = 0.0
            proj_x1 = proj_x2 = proj_y1 = proj_y2 = 0

        # Crack Rate CK (%)
        crack_rate_CK = (total_Q_mm2 / total_surface_area_mm2) * 100.0 if total_surface_area_mm2 > 0 else 0.0

        # PCI Score & Condition
        deduct_CK = min(50.0, crack_rate_CK * 15.0)
        deduct_width = min(50.0, (global_dmax / 1.5) * 40.0)
        pci_score = max(0.0, 100.0 - (deduct_CK + deduct_width))

        if pci_score >= 85.0:
            damage_level = "Good"
        elif pci_score >= 70.0:
            damage_level = "Satisfactory"
        elif pci_score >= 55.0:
            damage_level = "Fair"
        elif pci_score >= 40.0:
            damage_level = "Poor"
        elif pci_score >= 25.0:
            damage_level = "Severe"
        else:
            damage_level = "Critical"

        if global_dmax < 0.1:
            severity = "Hairline (<0.1 mm)"
            warning = "Negligible structural impact; cosmetic hairline monitoring."
        elif global_dmax < 0.3:
            severity = "Minor (0.1 - 0.3 mm)"
            warning = "Tolerable for interior dry structures; monitor for water penetration."
        elif global_dmax < 0.5:
            severity = "Moderate (0.3 - 0.5 mm)"
            warning = "Caution: exceeds typical exterior limits; potential moisture & corrosion risk."
        else:
            severity = "Severe (>0.5 mm)"
            warning = "HIGH RISK: Exceeds standard concrete crack thresholds; structural inspection required."

        return CrackMetrics(
            mu=self.mu,
            image_width=w,
            image_height=h,
            total_surface_area_mm2=round(total_surface_area_mm2, 2),
            total_unrefined_pixels_E=total_E,
            total_crack_area_Q_mm2=round(total_Q_mm2, 2),
            total_skeleton_length_L_mm=round(total_length_L_mm, 2),
            min_width_dmin_mm=round(global_dmin, 3),
            max_width_dmax_mm=round(global_dmax, 3),
            avg_width_davg_mm=round(global_davg, 3),
            proj_X1=proj_x1,
            proj_X2=proj_x2,
            proj_Y1=proj_y1,
            proj_Y2=proj_y2,
            mesh_bounding_area_S_mm2=round(mesh_area_S, 2),
            crack_rate_CK_percent=round(crack_rate_CK, 4),
            pavement_condition_index_PCI=round(pci_score, 1),
            damage_level=damage_level,
            severity_category=severity,
            structural_risk_warning=warning,
            components=component_list,
        )
