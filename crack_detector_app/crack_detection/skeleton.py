import cv2
import numpy as np


class SkeletonExtractor:
    """
    Crack skeleton extraction and refinement module.
    
    Direct implementation of the Paper Morphological Skeleton Algorithm (Eq 16-18):
    S(A) = U_{k=1}^K S_k(A)
    S_k(A) = (A (-) kB) - ((A (-) kB) o B)
    Runs in high-speed optimized C++ via OpenCV operators.
    """

    def __init__(self, method: str = "paper_morphological"):
        self.method = method

    def extract_paper_morphological_skeleton(
        self, binary_mask: np.ndarray
    ) -> np.ndarray:
        """
        Fast morphological skeleton extraction strictly following Section 3/4 Eq (16)-(18):
        S(A) = U_{k=1}^K [ (A (-) kB) - ((A (-) kB) o B) ]
        """
        if cv2.countNonZero(binary_mask) == 0:
            return np.zeros_like(binary_mask)

        img = (binary_mask > 0).astype(np.uint8) * 255
        skeleton = np.zeros_like(img, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        eroded = img.copy()
        max_iter = 50  # Crack fissures are typically single to dozens of pixels wide

        for _ in range(max_iter):
            if cv2.countNonZero(eroded) == 0:
                break
            # Opened = (A (-) kB) o B
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
            # Subset = (A (-) kB) - ((A (-) kB) o B)
            subset = cv2.subtract(eroded, opened)
            # S(A) = S(A) U S_k(A)
            skeleton = cv2.bitwise_or(skeleton, subset)
            # Consecutive erosion
            eroded = cv2.erode(eroded, element)

        return skeleton

    def extract_skeleton(self, binary_mask: np.ndarray) -> np.ndarray:
        return self.extract_paper_morphological_skeleton(binary_mask)
