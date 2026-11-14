import cv2
import numpy as np


class ImagePreprocessor:
    """
    Image preprocessing module for wall and floor crack detection using
    classical digital image processing techniques.
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.5,
        clahe_grid_size: int = 8,
        bilateral_d: int = 7,
        bilateral_sigma_color: float = 50.0,
        bilateral_sigma_space: float = 50.0,
        gaussian_ksize: int = 3,
        gaussian_sigma: float = 1.0,
    ):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.gaussian_ksize = gaussian_ksize
        self.gaussian_sigma = gaussian_sigma

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Converts an input BGR or RGB image to single-channel uint8 grayscale."""
        if len(image.shape) == 2:
            return image.copy()
        elif len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """
        Enhances local contrast and balances illumination across uneven wall surfaces
        using Contrast-Limited Adaptive Histogram Equalization.
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_grid_size, self.clahe_grid_size),
        )
        return clahe.apply(gray)

    def denoise_bilateral(self, gray: np.ndarray) -> np.ndarray:
        """
        Smooths surface grain/texture while strictly preserving sharp crack edges.
        """
        return cv2.bilateralFilter(
            gray,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space,
        )

    def denoise_gaussian(self, gray: np.ndarray) -> np.ndarray:
        """Applies Gaussian smoothing to suppress high-frequency random sensor noise."""
        if self.gaussian_ksize > 1:
            k = self.gaussian_ksize if self.gaussian_ksize % 2 == 1 else self.gaussian_ksize + 1
            return cv2.GaussianBlur(gray, (k, k), self.gaussian_sigma)
        return gray

    def preprocess(self, image: np.ndarray) -> dict:
        """
        Runs the complete preprocessing stage:
        1. Grayscale conversion
        2. Bilateral edge-preserving denoising
        3. CLAHE local contrast normalization
        4. Mild Gaussian smoothing
        """
        gray_raw = self.to_grayscale(image)
        denoised = self.denoise_bilateral(gray_raw)
        equalized = self.apply_clahe(denoised)
        smoothed = self.denoise_gaussian(equalized)

        return {
            "gray_raw": gray_raw,
            "denoised": denoised,
            "equalized": equalized,
            "preprocessed": smoothed,
        }
