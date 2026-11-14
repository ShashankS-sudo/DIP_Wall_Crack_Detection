import os
import io
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename

from crack_detection.pipeline import CrackDetector, PipelineResult, make_serializable
from crack_detection.exporter import ReportExporter

app = Flask(__name__)
app.config["SECRET_KEY"] = "dip-crack-detection-secret-2026"
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["REPORTS_FOLDER"] = os.path.join(os.path.dirname(__file__), "reports")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB limit

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["REPORTS_FOLDER"], exist_ok=True)

# Default detector instance
detector = CrackDetector(mu=0.92)


def ndarray_to_base64_jpeg(img: np.ndarray, quality: int = 90) -> str:
    """Encodes a BGR or Grayscale numpy array into a base64 JPEG string."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode(".jpg", img, encode_param)
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def detect_single():
    """
    Analyzes an uploaded image or existing sample image using classical DIP multiscale analysis.
    """
    try:
        # Check if file uploaded or filepath specified
        file = request.files.get("image")
        image_path = request.form.get("image_path")
        
        # Hyperparameters from form
        mu = float(request.form.get("mu", 0.92))
        otsu_factor = float(request.form.get("otsu_factor", 0.9))
        min_crack_area = int(request.form.get("min_crack_area", 40))
        threshold_mode = request.form.get("threshold_mode", "hybrid")
        draw_normals = request.form.get("draw_normals", "true").lower() == "true"
        draw_bboxes = request.form.get("draw_bboxes", "true").lower() == "true"

        img_bgr = None
        filename = "uploaded_image.jpg"

        if file and file.filename != "":
            filename = secure_filename(file.filename)
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif image_path and os.path.isfile(image_path):
            filename = os.path.basename(image_path)
            img_bgr = cv2.imread(image_path)
        else:
            return jsonify({"success": False, "error": "No valid image provided"}), 400

        if img_bgr is None:
            return jsonify({"success": False, "error": "Failed to decode image"}), 400

        # Configure custom detector instance if parameters changed
        custom_detector = CrackDetector(
            mu=mu,
            threshold_mode=threshold_mode,
            otsu_factor=otsu_factor,
            min_crack_area=min_crack_area,
        )

        result: PipelineResult = custom_detector.process_image(img_bgr, mu_override=mu)

        # Re-render overlay if drawing options altered
        overlay_bgr = custom_detector.visualizer.draw_measurement_overlay(
            original_bgr=result.original_bgr,
            crack_mask=result.binary_mask,
            skeleton_mask=result.skeleton_mask,
            metrics=result.metrics,
            draw_normals=draw_normals,
            draw_bboxes=draw_bboxes,
        )

        # Base64 representations for immediate reactive display in UI
        b64_overlay = ndarray_to_base64_jpeg(overlay_bgr)
        b64_grid = ndarray_to_base64_jpeg(result.comparison_grid_bgr)
        b64_original = ndarray_to_base64_jpeg(result.original_bgr)
        b64_mask = ndarray_to_base64_jpeg(result.binary_mask)
        b64_skeleton = ndarray_to_base64_jpeg(result.skeleton_mask)
        b64_preprocessed = ndarray_to_base64_jpeg(result.preprocessed_gray)
        b64_multiscale = ndarray_to_base64_jpeg(result.multiscale_response)

        data = result.to_dict()
        data["image_name"] = filename
        data["images"] = {
            "overlay": b64_overlay,
            "comparison_grid": b64_grid,
            "original": b64_original,
            "mask": b64_mask,
            "skeleton": b64_skeleton,
            "preprocessed": b64_preprocessed,
            "multiscale": b64_multiscale,
        }

        return jsonify(make_serializable({"success": True, "data": data}))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/batch_detect", methods=["POST"])
def batch_detect():
    """
    Runs batch crack detection over multiple uploaded files.
    """
    try:
        files = request.files.getlist("images")
        if not files or len(files) == 0:
            return jsonify({"success": False, "error": "No files provided"}), 400

        mu = float(request.form.get("mu", 0.92))
        otsu_factor = float(request.form.get("otsu_factor", 0.9))
        min_crack_area = int(request.form.get("min_crack_area", 40))

        batch_detector = CrackDetector(
            mu=mu,
            otsu_factor=otsu_factor,
            min_crack_area=min_crack_area,
        )

        batch_results = []
        for file in files:
            if not file.filename:
                continue
            filename = secure_filename(file.filename)
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            res = batch_detector.process_image(img_bgr, mu_override=mu)
            res_dict = res.to_dict()
            res_dict["image_name"] = filename
            # Include thumbnail overlay for UI table
            small_overlay = cv2.resize(res.overlay_bgr, (120, 90), interpolation=cv2.INTER_AREA)
            res_dict["thumbnail"] = ndarray_to_base64_jpeg(small_overlay, quality=70)
            batch_results.append(res_dict)

        # Export CSV report automatically
        csv_path = os.path.join(app.config["REPORTS_FOLDER"], "latest_batch_report.csv")
        ReportExporter.export_batch_csv(batch_results, csv_path)

        return jsonify(make_serializable({
            "success": True,
            "count": len(batch_results),
            "results": batch_results,
            "csv_ready": True,
        }))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/download_report_csv")
def download_report_csv():
    csv_path = os.path.join(app.config["REPORTS_FOLDER"], "latest_batch_report.csv")
    if os.path.isfile(csv_path):
        return send_file(csv_path, as_attachment=True, download_name="crack_detection_report.csv")
    return jsonify({"error": "Report not generated yet"}), 404


if __name__ == "__main__":
    print("=" * 70)
    print("  WALL & FLOOR CRACK DETECTION APPLICATION")
    print("  Based on: Building Crack Detection Based on DIP & Multiscale")
    print("  Mode: 100% Classical Digital Image Processing (Non-AI / Non-DL)")
    print("  Default Calibration Ratio: mu = 0.92 mm/pixel")
    print("  Web Server starting at: http://localhost:5000")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, debug=False)
