# Wall & Floor Crack Detection Application (Classical DIP & Multiscale)

> **Strict Non-AI / Non-Deep-Learning Implementation**  
> Based on the published research:  
> *"Building Crack Detection Based on Digital Image Processing Technology and Multiscale"* (Liu et al., *Advances in Multimedia*)

---

## 📌 Executive Overview

This application detects, segments, and quantitatively measures cracks on walls, floors, pavements, and concrete structures using **100% classical Digital Image Processing (DIP) and multiscale differential geometry**. 

**No neural networks, deep learning models, training weights, or AI inference engines are used.** Every detection step relies on verifiable mathematical operators:
1. **Multiscale Hessian Matrix Eigenvalue Analysis** (Frangi-Sato curvature line filtering)
2. **Multiscale Mathematical Morphology** (Black-Hat dark valley extraction across structuring element scales)
3. **Contrast-Limited Adaptive Histogram Equalization (CLAHE)** and Bilateral edge-preserving filtering
4. **Morphological Skeletonization & Thinning** (Equations 16–18 from the paper)
5. **Normal Distance Orthogonal Raycasting** for Crack Widths $d_{min}$, $d_{max}$, and $d_{avg}$ (Equations 19–21)
6. **Network / Mesh Crack Bounding Area Projection** (Equation 22)
7. **Pavement & Structural Condition Index (PCI) and Crack Rate ($CK$)**

---

## 📐 Mathematical Formulation (From the Paper)

| Parameter / Step | Formula from Report | Implementation in Code |
| :--- | :--- | :--- |
| **Skeleton Extraction** | $S(A) = \bigcup_{k=1}^K S_k(A)$ (Eq. 16)<br>$S_k(A) = (A \ominus kB) - ((A \ominus kB) \circ B)$ (Eq. 17)<br>$K = \max \{ k \mid A \ominus kB \neq \emptyset \}$ (Eq. 18) | `crack_detection/skeleton.py` via morphological erosion & opening subsets and Zhang-Suen medial axis thinning. |
| **Physical Length ($L$)** | $L = \mu \cdot l$ | `crack_detection/geometry.py`: $\mu = 0.92\text{ mm/pixel}$ by default (configurable), $l$ = skeleton pixel distance. |
| **Normal Line Width ($d$)** | $d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$ (Eq. 19)<br>$d_{min} = \min(d_i), \quad d_{max} = \max(d_i)$ (Eq. 20) | `compute_vertical_line_widths`: Casts normal rays perpendicular to crack tangent to find crack boundary intersection points $A(x_1, y_1)$ and $B(x_2, y_2)$. |
| **Crack Area ($Q$) & Avg Width ($d_{avg}$)** | $E = \sum_{x=1}^M \sum_{y=1}^N f(x, y)$ (Eq. 21)<br>$Q = \mu^2 \cdot E$<br>$d_{avg} = Q / L$ | `crack_detection/geometry.py`: $E$ = unrefined crack pixel count, $Q$ = true physical area in $\text{mm}^2$, $d_{avg} = Q / L$. |
| **Mesh / Network Area ($S$)** | $S = \mu^2 (X_2 - X_1) \times (Y_2 - Y_1)$ (Eq. 22) | Minimum circumscribed bounding projection on $X$ and $Y$ axes for branching/alligator cracks. |
| **Crack Rate ($CK$) & PCI** | $CK = \frac{Q}{\text{Surface Area}} \times 100\%$ | Evaluates structural condition: Good, Satisfactory, Fair, Poor, Severe, Critical (ACI 224R standards). |

---

## 🚀 How to Run the Application

You have **three ways** to run the application depending on your preference:

### Option 1: Web Dashboard (Recommended)
Offers a responsive interface with real-time sliders, interactive tabs, 6-stage diagnostic pipeline view, and batch CSV export.

```bash
cd d:\DIP_Project\crack_detector_app
python app.py
```
Then open your browser at: **`http://localhost:5000`**

#### Web Features:
- **Single Inspection Mode:** Drag & drop any wall/floor image, adjust calibration $\mu$ or sensitivity, view quantitative overlays.
- **6-Stage Pipeline Grid:** Side-by-side view of Original $\to$ Preprocessed $\to$ Multiscale Hessian $\to$ Binary Mask $\to$ Skeleton $\to$ Measurement Overlay.
- **Batch Processing Mode:** Select multiple images at once to analyze entire folders and download a generated CSV summary report.

---

### Option 2: Native Desktop GUI
A standalone desktop application window using Python Tkinter.

```bash
cd d:\DIP_Project\crack_detector_app
python gui.py
```

#### Desktop GUI Features:
- File browser dialog to select any image from disk.
- Live parameter sliders for calibration ratio $\mu$, threshold factor, and min crack area.
- Metric summary HUD strip (Length, Max Width, Avg Width, Area, Mesh Area, PCI Score).
- View switches between Overlay, 6-Stage Diagnostic Grid, and Skeleton Centerline.
- "Save Annotated Image" export button.

---

### Option 3: Command Line Interface (CLI)
Automated processing for single files or whole directories:

```bash
cd d:\DIP_Project\crack_detector_app

# Analyze a single image:
python cli.py --image "path\to\crack.jpg" --mu 0.92 --output-dir ./results

# Run batch analysis over an entire folder of images:
python cli.py --batch-dir "..\Wall Crack Image Dataset for Earthquake and Struct\Wall Crack Image Dataset" --output-dir ./results --csv dataset_report.csv
```

---

## 📂 Project Architecture

```
d:\DIP_Project\
├── Advances in Multimedia - 2022 - Liu - Building Crack Detection...pdf  # Reference paper
├── Wall Crack Image Dataset for Earthquake and Struct/                 # Dataset folder
├── README.md
└── crack_detector_app/
    ├── app.py                      # Modern Flask Web Application
    ├── gui.py                      # Standalone Native Desktop GUI
    ├── cli.py                      # Command-Line Automation Tool
    ├── templates/
    │   └── index.html              # Responsive Web Dashboard UI
    ├── static/                     # CSS & JS assets
    ├── reports/                    # Generated batch CSV & HTML reports
    └── crack_detection/            # Core Classical DIP Engine (Strictly Non-AI)
        ├── __init__.py
        ├── preprocessor.py         # CLAHE, Bilateral Denoising, Grayscale
        ├── multiscale.py           # Multiscale Hessian Eigenvalues & Black-Hat Morphology
        ├── segmentation.py         # Hybrid Otsu/Adaptive Thresholding & Component Filtering
        ├── skeleton.py             # Paper Morphological Skeleton (Eq 16-18) & Thinning
        ├── geometry.py             # Normal raycasting widths, Length, Area, Mesh, PCI (Eq 19-22)
        ├── visualizer.py           # Publication-ready overlays and 6-stage comparison grids
        ├── pipeline.py             # Unified CrackDetector & PipelineResult
        └── exporter.py             # CSV, JSON, and HTML inspection reports
```

---

## 🛡️ Non-AI / Non-Deep Learning Verification

To verify that **no AI or deep learning models** are present:
1. No `torch`, `torchvision`, `tensorflow`, `keras`, `onnxruntime`, `ultralytics`, or `transformers` packages are imported or required.
2. The code contains zero pretrained weights, checkpoints, or neural network layers.
3. Every step is purely based on classical numerical matrix operations in OpenCV and NumPy.
