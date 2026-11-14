import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Add current folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crack_detection.pipeline import CrackDetector, PipelineResult
from crack_detection.exporter import ReportExporter


class CrackDetectionDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Building Crack Detection System (DIP Multiscale - Non-AI)")
        self.root.geometry("1240x800")
        self.root.minsize(1000, 680)

        self.detector = CrackDetector(mu=0.92)
        self.current_image_path = None
        self.current_result: PipelineResult = None
        self.display_mode = "overlay"  # "overlay", "grid", "original", "skeleton"

        self._build_ui()

    def _build_ui(self):
        # Configure styles
        style = ttk.Style()
        style.theme_use("clam")

        # Top Header Frame
        header = tk.Frame(self.root, bg="#0f172a", height=60)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        title_lbl = tk.Label(
            header,
            text="Wall & Floor Crack Detection System",
            fg="#f8fafc",
            bg="#0f172a",
            font=("Segoe UI", 14, "bold"),
        )
        title_lbl.pack(side=tk.LEFT, padx=18, pady=5)

        subtitle_lbl = tk.Label(
            header,
            text="Pure Digital Image Processing & Multiscale Feature Analysis (No AI / No DL)",
            fg="#38bdf8",
            bg="#0f172a",
            font=("Segoe UI", 9, "italic"),
        )
        subtitle_lbl.pack(side=tk.LEFT, padx=5, pady=5)

        # Main Layout: Left Sidebar + Right Display
        main_frame = tk.Frame(self.root, bg="#f1f5f9")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Sidebar (Controls)
        sidebar = tk.Frame(main_frame, bg="#ffffff", width=340, relief=tk.RIDGE, bd=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar.pack_propagate(False)

        # File Selection Box
        lbl_file_sec = tk.Label(
            sidebar, text="1. Select Image", font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#0f172a"
        )
        lbl_file_sec.pack(anchor=tk.W, padx=15, pady=(15, 5))

        btn_browse = tk.Button(
            sidebar,
            text="📁 Browse Image File...",
            command=self.browse_image,
            bg="#0284c7",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=6,
            cursor="hand2",
        )
        btn_browse.pack(fill=tk.X, padx=15, pady=4)

        self.lbl_selected_file = tk.Label(
            sidebar, text="No image selected", fg="#64748b", bg="#ffffff", font=("Segoe UI", 8), wraplength=300
        )
        self.lbl_selected_file.pack(anchor=tk.W, padx=15, pady=2)

        # Separator
        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=15, pady=12)

        # Parameters Box
        lbl_param_sec = tk.Label(
            sidebar, text="2. DIP Parameters (Paper Eq 16-22)", font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#0f172a"
        )
        lbl_param_sec.pack(anchor=tk.W, padx=15, pady=(0, 5))

        # Mu Calibration
        lbl_mu = tk.Label(sidebar, text="Calibration μ (mm/px):", bg="#ffffff", font=("Segoe UI", 9))
        lbl_mu.pack(anchor=tk.W, padx=15, pady=(4, 0))
        self.entry_mu = ttk.Entry(sidebar)
        self.entry_mu.insert(0, "0.92")
        self.entry_mu.pack(fill=tk.X, padx=15, pady=2)

        # Sensitivity (Otsu factor)
        lbl_sens = tk.Label(sidebar, text="Threshold Factor (0.5 - 1.3):", bg="#ffffff", font=("Segoe UI", 9))
        lbl_sens.pack(anchor=tk.W, padx=15, pady=(4, 0))
        self.slider_sens = ttk.Scale(sidebar, from_=0.5, to=1.3, value=0.9, orient=tk.HORIZONTAL)
        self.slider_sens.pack(fill=tk.X, padx=15, pady=2)

        # Min Crack Area
        lbl_area = tk.Label(sidebar, text="Min Crack Area (pixels):", bg="#ffffff", font=("Segoe UI", 9))
        lbl_area.pack(anchor=tk.W, padx=15, pady=(4, 0))
        self.slider_area = ttk.Scale(sidebar, from_=10, to=200, value=40, orient=tk.HORIZONTAL)
        self.slider_area.pack(fill=tk.X, padx=15, pady=2)

        # Threshold Mode
        lbl_mode = tk.Label(sidebar, text="Segmentation Mode:", bg="#ffffff", font=("Segoe UI", 9))
        lbl_mode.pack(anchor=tk.W, padx=15, pady=(4, 0))
        self.combo_mode = ttk.Combobox(
            sidebar, values=["hybrid", "hysteresis", "otsu", "adaptive"], state="readonly"
        )
        self.combo_mode.set("hybrid")
        self.combo_mode.pack(fill=tk.X, padx=15, pady=2)

        # Action Button
        btn_detect = tk.Button(
            sidebar,
            text="⚡ Run Crack Detection",
            command=self.run_detection,
            bg="#0f172a",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=8,
            cursor="hand2",
        )
        btn_detect.pack(fill=tk.X, padx=15, pady=(18, 5))

        # Separator
        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=15, pady=12)

        # View Mode Switches
        lbl_view_sec = tk.Label(
            sidebar, text="3. View Modes", font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#0f172a"
        )
        lbl_view_sec.pack(anchor=tk.W, padx=15, pady=(0, 5))

        btn_view_overlay = tk.Button(
            sidebar, text="🔍 Overlay & Measurements", command=lambda: self.set_view("overlay"), relief=tk.GROOVE
        )
        btn_view_overlay.pack(fill=tk.X, padx=15, pady=2)

        btn_view_grid = tk.Button(
            sidebar, text="📊 6-Stage Diagnostic Grid", command=lambda: self.set_view("grid"), relief=tk.GROOVE
        )
        btn_view_grid.pack(fill=tk.X, padx=15, pady=2)

        btn_view_skeleton = tk.Button(
            sidebar, text="🕸️ 1-px Skeleton (Eq 16-18)", command=lambda: self.set_view("skeleton"), relief=tk.GROOVE
        )
        btn_view_skeleton.pack(fill=tk.X, padx=15, pady=2)

        btn_export = tk.Button(
            sidebar,
            text="💾 Save Annotated Image...",
            command=self.save_output_image,
            bg="#10b981",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            pady=4,
        )
        btn_export.pack(fill=tk.X, padx=15, pady=(12, 5))

        # Right Panel (Metrics + Image Canvas)
        right_panel = tk.Frame(main_frame, bg="#f1f5f9")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Top Metrics Strip
        self.metrics_frame = tk.Frame(right_panel, bg="#ffffff", relief=tk.RIDGE, bd=1, height=90)
        self.metrics_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 10))
        self.metrics_frame.pack_propagate(False)

        self._init_metric_widgets()

        # Canvas for displaying image
        self.canvas_frame = tk.Frame(right_panel, bg="#020617")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#020617", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _init_metric_widgets(self):
        f = self.metrics_frame
        col_w = 120

        def make_metric(col, label_text, default_val):
            lbl_title = tk.Label(f, text=label_text, fg="#64748b", bg="#ffffff", font=("Segoe UI", 8, "bold"))
            lbl_title.grid(row=0, column=col, padx=10, pady=(10, 2), sticky="w")
            lbl_val = tk.Label(f, text=default_val, fg="#0f172a", bg="#ffffff", font=("Segoe UI", 12, "bold"))
            lbl_val.grid(row=1, column=col, padx=10, pady=(0, 10), sticky="w")
            return lbl_val

        self.m_length = make_metric(0, "LENGTH (L)", "-- mm")
        self.m_max_w = make_metric(1, "MAX WIDTH (dmax)", "-- mm")
        self.m_avg_w = make_metric(2, "AVG WIDTH (davg)", "-- mm")
        self.m_area = make_metric(3, "CRACK AREA (Q)", "-- mm²")
        self.m_mesh = make_metric(4, "MESH AREA (S)", "-- mm²")
        self.m_pci = make_metric(5, "PCI SCORE", "-- / 100")
        self.m_status = make_metric(6, "CONDITION", "Ready")

    def browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Wall/Floor Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.current_image_path = path
            self.lbl_selected_file.config(text=os.path.basename(path))
            # Load and display raw image
            img = cv2.imread(path)
            if img is not None:
                self.show_cv2_image(img)

    def run_detection(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please select an image file first!")
            return

        try:
            mu = float(self.entry_mu.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Calibration μ must be a valid number!")
            return

        otsu_factor = float(self.slider_sens.get())
        min_crack_area = int(self.slider_area.get())
        thresh_mode = self.combo_mode.get()

        custom_detector = CrackDetector(
            mu=mu,
            threshold_mode=thresh_mode,
            otsu_factor=otsu_factor,
            min_crack_area=min_crack_area,
        )

        try:
            self.current_result = custom_detector.process_image(self.current_image_path, mu_override=mu)
            self.update_metrics_display()
            self.set_view(self.display_mode)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")

    def update_metrics_display(self):
        if not self.current_result:
            return
        m = self.current_result.metrics
        self.m_length.config(text=f"{m.total_skeleton_length_L_mm:.1f} mm")
        self.m_max_w.config(text=f"{m.max_width_dmax_mm:.2f} mm")
        self.m_avg_w.config(text=f"{m.avg_width_davg_mm:.2f} mm")
        self.m_area.config(text=f"{m.total_crack_area_Q_mm2:.1f} mm²")
        self.m_mesh.config(text=f"{m.mesh_bounding_area_S_mm2:.1f} mm²")
        self.m_pci.config(text=f"{m.pavement_condition_index_PCI:.0f}")
        self.m_status.config(text=f"{m.damage_level}")

    def set_view(self, mode: str):
        self.display_mode = mode
        if not self.current_result:
            return
        if mode == "overlay":
            self.show_cv2_image(self.current_result.overlay_bgr)
        elif mode == "grid":
            self.show_cv2_image(self.current_result.comparison_grid_bgr)
        elif mode == "skeleton":
            # Convert single channel skeleton to 3-channel for rendering
            skel_bgr = cv2.cvtColor(self.current_result.skeleton_mask, cv2.COLOR_GRAY2BGR)
            self.show_cv2_image(skel_bgr)

    def show_cv2_image(self, bgr_img: np.ndarray):
        # Resize to fit canvas
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 100 or canvas_h < 100:
            canvas_w, canvas_h = 800, 600

        img_h, img_w = bgr_img.shape[:2]
        ratio = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        new_w, new_h = max(1, int(img_w * ratio)), max(1, int(img_h * ratio))

        resized = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        self.tk_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        # Center image
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_image)

    def save_output_image(self):
        if not self.current_result:
            messagebox.showinfo("No Output", "Run detection first before saving.")
            return

        out_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            title="Save Output Inspection Image",
        )
        if out_path:
            cv2.imwrite(out_path, self.current_result.overlay_bgr)
            messagebox.showinfo("Saved", f"Annotated image saved successfully to:\n{out_path}")


def main():
    root = tk.Tk()
    app = CrackDetectionDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
