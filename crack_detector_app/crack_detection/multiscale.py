import cv2
import numpy as np
from typing import List, Tuple, Dict


class MultiscaleFeatureExtractor:
    """
    Multiscale feature analysis for building and floor crack detection.
    
    Implements classical multiscale differential geometry and multiscale
    mathematical morphology without any neural networks or AI models:
    1. Multiscale Hessian Matrix (Eigenvalue analysis / Frangi line filter)
    2. Multiscale Morphological Black-Hat (dark valley detection across scales)
    3. Multiscale Directional Edge Gradients
    """

    def __init__(
        self,
        scales: List[float] = None,
        beta: float = 0.5,
        c_factor: float = 0.5,
        morph_scales: List[int] = None,
        weight_hessian: float = 0.65,
        weight_morph: float = 0.35,
    ):
        # Continuous scales for fine hairline cracks up to wide fissures
        self.scales = scales if scales is not None else [1.0, 1.8, 2.8, 4.0, 6.0]
        self.beta = beta
        self.c_factor = c_factor
        # Structuring element kernel sizes for morphological multiscale
        self.morph_scales = morph_scales if morph_scales is not None else [3, 7, 13, 21]
        self.weight_hessian = weight_hessian
        self.weight_morph = weight_morph

    def _compute_hessian_eigenvalues(
        self, image: np.ndarray, sigma: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the eigenvalues of the Hessian matrix at scale sigma:
        H = [ Lxx  Lxy ]
            [ Lyx  Lyy ]
        where L is Gaussian-smoothed image at scale sigma.
        """
        # Ensure image is float in [0, 1]
        img_float = image.astype(np.float32) / 255.0

        # Gaussian smoothing kernel size
        ksize = int(2 * np.ceil(3 * sigma) + 1)
        smoothed = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)

        # First derivatives using Scharr/Sobel
        Lx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        Ly = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)

        # Second derivatives with scale normalization (sigma^2)
        scale_factor = sigma ** 2
        Lxx = cv2.Sobel(Lx, cv2.CV_32F, 1, 0, ksize=3) * scale_factor
        Lyy = cv2.Sobel(Ly, cv2.CV_32F, 0, 1, ksize=3) * scale_factor
        Lxy = cv2.Sobel(Lx, cv2.CV_32F, 0, 1, ksize=3) * scale_factor

        # Eigenvalues calculation for 2x2 symmetric matrix
        trace = Lxx + Lyy
        det = Lxx * Lyy - (Lxy ** 2)

        # Discriminant of characteristic polynomial: trace^2 - 4*det
        disc = np.sqrt(np.maximum(0.0, (trace ** 2) - 4.0 * det))

        lambda_1_raw = 0.5 * (trace + disc)
        lambda_2_raw = 0.5 * (trace - disc)

        # Sort so that |lambda1| <= |lambda2|
        abs1 = np.abs(lambda_1_raw)
        abs2 = np.abs(lambda_2_raw)

        swap_mask = abs1 > abs2
        lambda1 = np.where(swap_mask, lambda_2_raw, lambda_1_raw)
        lambda2 = np.where(swap_mask, lambda_1_raw, lambda_2_raw)

        return lambda1, lambda2

    def frangi_line_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Multiscale Hessian-based ridge/valley filter.
        For dark crack lines on brighter concrete/wall surfaces, the cross-sectional
        curvature lambda2 is positive and dominates lambda1.
        """
        max_response = np.zeros_like(image, dtype=np.float32)
        eps = 1e-6

        # Estimate sensitivity threshold c from gradient Frobenius norm
        for sigma in self.scales:
            lambda1, lambda2 = self._compute_hessian_eigenvalues(image, sigma)

            # Ratio of blobness: Rb = |lambda1| / |lambda2|
            rb = np.abs(lambda1) / (np.abs(lambda2) + eps)

            # Second order structureness (Frobenius norm)
            s_squared = (lambda1 ** 2) + (lambda2 ** 2)
            c = self.c_factor * 0.5 * (np.max(s_squared) + eps)

            # Line / Valley filter for dark crack lines:
            # High response when lambda2 > 0 (valley profile) and rb is small (linear, not blob)
            term_blob = np.exp(-(rb ** 2) / (2.0 * (self.beta ** 2)))
            term_structure = 1.0 - np.exp(-s_squared / (2.0 * c + eps))

            response = term_blob * term_structure
            # Suppress non-valley regions (where curvature across ridge is not positive)
            response[lambda2 <= 0] = 0.0

            max_response = np.maximum(max_response, response)

        # Normalize to [0, 255] uint8
        norm = cv2.normalize(max_response, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return norm.astype(np.uint8)

    def multiscale_morphological_blackhat(self, image: np.ndarray) -> np.ndarray:
        """
        Multiscale Black-Hat (Bottom-Hat) transform.
        Extracts narrow dark troughs and fissures of varying widths embedded in
        textured or non-uniform background surfaces.
        """
        accumulated_response = np.zeros_like(image, dtype=np.float32)

        for size in self.morph_scales:
            # Elliptic / cross-structuring element models crack fissures well
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
            accumulated_response += blackhat.astype(np.float32)

        # Average and normalize
        norm = cv2.normalize(
            accumulated_response, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )
        return norm.astype(np.uint8)

    def extract_multiscale_features(self, preprocessed_gray: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Executes multiscale feature analysis combining differential geometry
        (Hessian line filter) and multiscale mathematical morphology.
        """
        hessian_response = self.frangi_line_filter(preprocessed_gray)
        morph_response = self.multiscale_morphological_blackhat(preprocessed_gray)

        # Weighted fusion of multiscale feature maps
        fused = (
            self.weight_hessian * hessian_response.astype(np.float32)
            + self.weight_morph * morph_response.astype(np.float32)
        )
        fused_norm = np.clip(fused, 0, 255).astype(np.uint8)

        # Multiscale gradient magnitude for edge enhancement
        sobel_x = cv2.Sobel(fused_norm, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(fused_norm, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(sobel_x, sobel_y)
        grad_norm = cv2.normalize(grad_mag, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

        return {
            "hessian_response": hessian_response,
            "morph_response": morph_response,
            "fused_multiscale": fused_norm,
            "gradient_enhanced": grad_norm,
        }
