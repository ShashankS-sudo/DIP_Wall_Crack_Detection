import csv
import json
import os
from typing import List, Dict, Any
from .pipeline import PipelineResult


class ReportExporter:
    """
    Exports crack inspection and detection data to CSV, JSON, and HTML inspection reports.
    """

    @staticmethod
    def export_batch_csv(results: List[Dict[str, Any]], output_filepath: str):
        """Exports batch detection metrics to a CSV file."""
        if not results:
            return

        fieldnames = [
            "image_name",
            "calibration_mu_mm_px",
            "crack_detected",
            "num_crack_components",
            "skeleton_length_L_mm",
            "crack_area_Q_mm2",
            "max_width_dmax_mm",
            "avg_width_davg_mm",
            "min_width_dmin_mm",
            "mesh_area_S_mm2",
            "crack_rate_CK_pct",
            "pci_score",
            "damage_level",
            "severity_category",
        ]

        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)

        with open(output_filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {
                    "image_name": r.get("image_name", ""),
                    "calibration_mu_mm_px": r.get("calibration_mu_mm_per_px", 0.92),
                    "crack_detected": r.get("crack_measurements", {}).get("skeleton_length_L_mm", 0) > 0,
                    "num_crack_components": r.get("component_count", 0),
                    "skeleton_length_L_mm": r.get("crack_measurements", {}).get("skeleton_length_L_mm", 0),
                    "crack_area_Q_mm2": r.get("crack_measurements", {}).get("crack_area_Q_mm2", 0),
                    "max_width_dmax_mm": r.get("crack_measurements", {}).get("max_width_dmax_mm", 0),
                    "avg_width_davg_mm": r.get("crack_measurements", {}).get("avg_width_davg_mm", 0),
                    "min_width_dmin_mm": r.get("crack_measurements", {}).get("min_width_dmin_mm", 0),
                    "mesh_area_S_mm2": r.get("network_crack_mesh", {}).get("mesh_bounding_area_S_mm2", 0),
                    "crack_rate_CK_pct": r.get("condition_assessment", {}).get("crack_rate_CK_percent", 0),
                    "pci_score": r.get("condition_assessment", {}).get("pavement_condition_index_PCI", 100),
                    "damage_level": r.get("condition_assessment", {}).get("damage_level", "Good"),
                    "severity_category": r.get("condition_assessment", {}).get("severity_category", "None"),
                }
                writer.writerow(row)

    @staticmethod
    def export_inspection_html(result: PipelineResult, image_name: str, output_filepath: str):
        """Generates an executive single-image inspection report in clean HTML."""
        m = result.metrics
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Crack Inspection Report - {image_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background: #f8fafc; color: #1e293b; }}
        .card {{ background: #ffffff; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 24px; }}
        h1 {{ margin-top: 0; color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 14px; }}
        .badge-severe {{ background: #fee2e2; color: #991b1b; }}
        .badge-mod {{ background: #ffedd5; color: #9a3412; }}
        .badge-minor {{ background: #fef9c3; color: #854d0e; }}
        .badge-hairline {{ background: #dcfce7; color: #166534; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f1f5f9; font-weight: 600; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
        .stat-card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }}
        .stat-val {{ font-size: 22px; font-weight: bold; color: #0284c7; margin-top: 4px; }}
        .stat-label {{ font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Building Crack Inspection Report</h1>
        <p><strong>Inspected Image:</strong> {image_name} | <strong>Algorithm:</strong> Classical Multiscale Digital Image Processing (Non-AI)</p>
        <p><strong>Calibration Factor (&mu;):</strong> {m.mu} mm/pixel | <strong>Pavement Condition Index (PCI):</strong> {m.pavement_condition_index_PCI}/100 ({m.damage_level})</p>
        <p><strong>Severity Status:</strong> <span class="badge badge-{'severe' if m.max_width_dmax_mm>=0.5 else 'mod' if m.max_width_dmax_mm>=0.3 else 'minor' if m.max_width_dmax_mm>=0.1 else 'hairline'}">{m.severity_category}</span></p>
        <p><strong>Engineering Assessment:</strong> {m.structural_risk_warning}</p>
    </div>

    <div class="card">
        <h2>Quantitative Crack Metrics (Section 3/4, Eq 16-22)</h2>
        <div class="grid">
            <div class="stat-card">
                <div class="stat-label">Total Crack Length (L)</div>
                <div class="stat-val">{m.total_skeleton_length_L_mm:.1f} mm</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Maximum Width (d_max)</div>
                <div class="stat-val">{m.max_width_dmax_mm:.2f} mm</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average Width (d_avg)</div>
                <div class="stat-val">{m.avg_width_davg_mm:.2f} mm</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Physical Crack Area (Q)</div>
                <div class="stat-val">{m.total_crack_area_Q_mm2:.1f} mm&sup2;</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mesh Bounding Area (S)</div>
                <div class="stat-val">{m.mesh_bounding_area_S_mm2:.1f} mm&sup2;</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Crack Rate (CK)</div>
                <div class="stat-val">{m.crack_rate_CK_percent:.3f}%</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Detected Crack Components Breakdown</h2>
        <table>
            <thead>
                <tr>
                    <th>Component #</th>
                    <th>Length L (mm)</th>
                    <th>Max Width d_max (mm)</th>
                    <th>Avg Width d_avg (mm)</th>
                    <th>Area Q (mm&sup2;)</th>
                    <th>Mesh Area S (mm&sup2;)</th>
                    <th>Type</th>
                </tr>
            </thead>
            <tbody>
                {''.join([f"<tr><td>#{c.component_id}</td><td>{c.length_L_mm}</td><td>{c.max_width_dmax_mm}</td><td>{c.avg_width_davg_mm}</td><td>{c.actual_area_Q_mm2}</td><td>{c.mesh_area_S_mm2}</td><td>{'Network/Mesh' if c.is_network_crack else 'Linear'}</td></tr>" for c in m.components])}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(html)
