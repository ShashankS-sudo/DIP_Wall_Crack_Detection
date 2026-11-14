import os
import sys
import argparse
import glob
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crack_detection.pipeline import CrackDetector
from crack_detection.exporter import ReportExporter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classical DIP Building Crack Detection CLI (Strict Non-AI / Non-DL)"
    )
    parser.add_argument("--image", type=str, help="Path to single input image file")
    parser.add_argument("--batch-dir", type=str, help="Directory containing images for batch analysis")
    parser.add_argument("--output-dir", type=str, default="./output_reports", help="Directory to save output results")
    parser.add_argument("--mu", type=float, default=0.92, help="Physical calibration factor in mm/pixel (default: 0.92)")
    parser.add_argument("--otsu-factor", type=float, default=0.9, help="Sensitivity threshold factor (default: 0.9)")
    parser.add_argument("--min-area", type=int, default=40, help="Min connected crack component size in px (default: 40)")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "hysteresis", "otsu", "adaptive"])
    parser.add_argument("--csv", type=str, default="batch_metrics.csv", help="CSV filename for batch export")
    parser.add_argument("--save-grids", action="store_true", help="Save 6-stage diagnostic pipeline comparison grids")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    detector = CrackDetector(
        mu=args.mu,
        otsu_factor=args.otsu_factor,
        min_crack_area=args.min_area,
        threshold_mode=args.mode,
    )

    if args.image:
        if not os.path.isfile(args.image):
            print(f"Error: Image file not found: {args.image}")
            sys.exit(1)

        print(f"[+] Processing single image: {args.image} (mu = {args.mu} mm/px)")
        res = detector.process_image(args.image, mu_override=args.mu)
        base = os.path.splitext(os.path.basename(args.image))[0]

        # Save annotated overlay
        out_overlay = os.path.join(args.output_dir, f"{base}_annotated.jpg")
        cv2.imwrite(out_overlay, res.overlay_bgr)
        print(f"[+] Saved annotated overlay: {out_overlay}")

        # Save diagnostic grid
        out_grid = os.path.join(args.output_dir, f"{base}_pipeline_stages.jpg")
        cv2.imwrite(out_grid, res.comparison_grid_bgr)
        print(f"[+] Saved 6-stage comparison: {out_grid}")

        # Save HTML report
        out_html = os.path.join(args.output_dir, f"{base}_report.html")
        ReportExporter.export_inspection_html(res, os.path.basename(args.image), out_html)
        print(f"[+] Saved inspection HTML: {out_html}")

        # Print metrics
        m = res.metrics
        print("\n" + "=" * 60)
        print("          DIP QUANTITATIVE INSPECTION SUMMARY          ")
        print("=" * 60)
        print(f"Crack Detected?        : {'YES' if m.total_unrefined_pixels_E > 0 else 'NO'}")
        print(f"Skeleton Length (L)    : {m.total_skeleton_length_L_mm:.2f} mm")
        print(f"Max Crack Width (dmax) : {m.max_width_dmax_mm:.2f} mm")
        print(f"Avg Crack Width (davg) : {m.avg_width_davg_mm:.2f} mm")
        print(f"Min Crack Width (dmin) : {m.min_width_dmin_mm:.2f} mm")
        print(f"Crack Area (Q)         : {m.total_crack_area_Q_mm2:.2f} mm² (Unrefined Pixels E = {m.total_unrefined_pixels_E})")
        print(f"Mesh Bounding Area (S) : {m.mesh_bounding_area_S_mm2:.2f} mm²")
        print(f"Crack Rate (CK)        : {m.crack_rate_CK_percent:.4f} %")
        print(f"Condition Index (PCI)  : {m.pavement_condition_index_PCI:.1f} / 100 ({m.damage_level})")
        print(f"Severity Category      : {m.severity_category}")
        print(f"Engineering Risk       : {m.structural_risk_warning}")
        print("=" * 60 + "\n")

    elif args.batch_dir:
        if not os.path.isdir(args.batch_dir):
            print(f"Error: Batch directory not found: {args.batch_dir}")
            sys.exit(1)

        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(args.batch_dir, ext)))
            image_files.extend(glob.glob(os.path.join(args.batch_dir, "**", ext), recursive=True))

        image_files = sorted(list(set(image_files)))
        print(f"[+] Found {len(image_files)} images in {args.batch_dir}")
        if len(image_files) == 0:
            print("No images found in specified directory.")
            sys.exit(0)

        results = []
        for fpath in image_files:
            try:
                res = detector.process_image(fpath, mu_override=args.mu)
                d = res.to_dict()
                d["image_name"] = os.path.basename(fpath)
                results.append(d)

                base = os.path.splitext(os.path.basename(fpath))[0]
                out_overlay = os.path.join(args.output_dir, f"{base}_annotated.jpg")
                cv2.imwrite(out_overlay, res.overlay_bgr)

                if args.save_grids:
                    out_grid = os.path.join(args.output_dir, f"{base}_pipeline_stages.jpg")
                    cv2.imwrite(out_grid, res.comparison_grid_bgr)

            except Exception as e:
                print(f"[-] Error processing {fpath}: {e}")

        csv_path = os.path.join(args.output_dir, args.csv)
        ReportExporter.export_batch_csv(results, csv_path)
        print(f"\n[+] Batch processing complete! Results saved to: {csv_path}")

    else:
        print("Please provide either --image or --batch-dir. Run with --help for options.")


if __name__ == "__main__":
    main()
