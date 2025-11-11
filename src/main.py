# src/main.py
"""
Turret-Tuner main entrypoint.

Provides a CLI and orchestration for:
 - running a web UI (Streamlit recommended)
 - running a headless analysis pipeline (load -> analyze -> identify -> tune -> report)
 - generating PDF reports

Usage examples (after installing requirements):
  python -m src.main analyze --use-demo
  python -m src.main analyze --input sample_data/demo.csv
  python -m src.main web
  python -m src.main report --analysis-result-path some_result.json
"""

from __future__ import annotations

import sys
import os
import logging
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import typer

# Attempt to import project modules; if missing, provide informative fallbacks.
try:
    from src.utils.data_loader import load_csv
except Exception:  # pragma: no cover - fallback behavior
    load_csv = None  # type: ignore

try:
    from src.analysis.analyzer import run_analysis
except Exception:  # pragma: no cover - fallback behavior
    run_analysis = None  # type: ignore

try:
    from src.system_id.system_identification import identify_system
except Exception:  # pragma: no cover - fallback behavior
    identify_system = None  # type: ignore

try:
    from src.optimization.tuner import tune_controller
except Exception:  # pragma: no cover - fallback behavior
    tune_controller = None  # type: ignore

try:
    # Optional: a report generator module if you implement one separately
    from src.utils.report_generator import generate_pdf_report
except Exception:
    generate_pdf_report = None  # type: ignore

# Third-party libs for fallback PDF/plots if needed
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import letter
except Exception:
    SimpleDocTemplate = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None  # type: ignore

app = typer.Typer(help="Turret-Tuner: analysis, tuning, and report generation CLI")

# Basic logging configuration
LOG = logging.getLogger("turret_tuner")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
LOG.addHandler(handler)
LOG.setLevel(logging.INFO)


# Project paths
ROOT = Path(__file__).resolve().parents[1]  # points to project root (Turret-Tuner/)
SRC = ROOT / "src"
SAMPLE_DATA = ROOT / "sample_data"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_module_available(name: str, obj: Any) -> None:
    if obj is None:
        LOG.error(
            "Required module/function '%s' not available. "
            "Install or add the file at the expected path.", name
        )
        raise RuntimeError(f"Missing required module/function: {name}")


def _timestamped_report_path(prefix: str = "turret_report", ext: str = "pdf") -> Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return REPORTS_DIR / f"{prefix}_{ts}.{ext}"


def fallback_demo_analysis(csv_path: Path) -> Dict[str, Any]:
    """
    Minimal demo analysis if analyzer module not present.
    Produces a fake result dict used for demoing the UI and PDF generator.
    """
    LOG.warning("Using fallback demo analysis (real analyzer module not found).")
    # Try to read CSV to produce some numbers, if possible
    data_info = {}
    try:
        import pandas as pd  # local import to avoid hard dependency if unused
        df = pd.read_csv(csv_path)
        data_info["rows"] = len(df)
        data_info["columns"] = list(df.columns)
    except Exception:
        data_info["rows"] = None
        data_info["columns"] = None

    # Fake Bayesian "results"
    demo_results = {
        "input_file": str(csv_path),
        "data_summary": data_info,
        "system_parameters": {
            "kV_mean": 0.12,
            "kV_std": 0.005,
            "kA_mean": 0.01,
            "kA_std": 0.002,
            "friction": 0.05,
        },
        "posterior_samples": None,
        "diagnostics": {"rmse": 0.034, "r2": 0.97},
        "tuned_gains": {"P": 0.6, "I": 0.02, "D": 0.005, "FF": 0.11},
        "notes": "Demo analysis: replace with real analyzer for true results.",
    }
    LOG.info("Demo analysis complete.")
    return demo_results


def fallback_generate_pdf(result: Dict[str, Any], out_path: Path) -> Path:
    """
    Basic PDF generator using reportlab and matplotlib as a fallback if
    src.utils.report_generator.generate_pdf_report does not exist.
    Produces a simple document with key fields and a tiny demo plot.
    """
    LOG.info("Generating fallback PDF report at %s", out_path)
    if SimpleDocTemplate is None:
        LOG.error("reportlab not installed; cannot generate PDF fallback.")
        raise RuntimeError("reportlab required for PDF generation fallback.")

    doc = SimpleDocTemplate(str(out_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Turret-Tuner Analysis Report (DEMO)", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.utcnow().isoformat()} UTC", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Input Data:", styles["Heading2"]))
    story.append(Paragraph(str(result.get("input_file", "unknown")), styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("System Parameters (summary):", styles["Heading2"]))
    params = result.get("system_parameters", {})
    table_data = [["Parameter", "Value"]]
    for k, v in params.items():
        table_data.append([k, str(v)])
    story.append(Table(table_data))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Tuned Gains:", styles["Heading2"]))
    gains = result.get("tuned_gains", {})
    table_data = [["Gain", "Value"]]
    for k, v in gains.items():
        table_data.append([k, str(v)])
    story.append(Table(table_data))
    story.append(Spacer(1, 12))

    # Optional tiny plot if matplotlib is available
    if plt is not None:
        try:
            fig_path = REPORTS_DIR / "demo_plot.png"
            # create a tiny dummy plot
            plt.figure(figsize=(4, 2.5))
            plt.title("Demo: measured vs predicted (fake)")
            plt.plot([0, 1, 2, 3, 4], [0, 0.9, 1.8, 2.6, 3.2], label="measured")
            plt.plot([0, 1, 2, 3, 4], [0, 1.0, 2.0, 3.0, 4.0], "--", label="predicted")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            story.append(Image(str(fig_path)))
            story.append(Spacer(1, 12))
        except Exception as e:
            LOG.warning("Could not generate demo plot: %s", e)

    story.append(Paragraph("Diagnostics:", styles["Heading2"]))
    diag = result.get("diagnostics", {})
    story.append(Paragraph(json.dumps(diag), styles["Code"]))
    doc.build(story)
    LOG.info("Fallback PDF generated.")
    return out_path


def orchestrate_pipeline(
    input_csv: Path,
    output_pdf_path: Optional[Path] = None,
    save_json: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrates the whole pipeline:
     1. Load data
     2. Run Bayesian analysis (or fallback demo)
     3. Perform system identification
     4. Tune controller
     5. Generate and save PDF report (returns path in 'report_path')

    Returns result dict containing all key artifacts (and path to saved PDF).
    """
    LOG.info("Starting pipeline with input: %s", input_csv)

    # 1) load data (best-effort)
    data = None
    if load_csv:
        try:
            LOG.debug("Loading CSV via src.utils.data_loader.load_csv")
            data = load_csv(input_csv)
        except Exception as exc:
            LOG.warning("load_csv failed: %s. Proceeding without loaded DataFrame.", exc)
            data = None
    else:
        LOG.info("No data_loader available; skipping explicit CSV load.")

    # 2) Run analyzer
    if run_analysis:
        try:
            LOG.debug("Calling src.analysis.analyzer.run_analysis")
            analysis_result = run_analysis(input_csv)
        except Exception as exc:
            LOG.exception("Analyzer failed: %s. Falling back to demo analyzer.", exc)
            analysis_result = fallback_demo_analysis(input_csv)
    else:
        analysis_result = fallback_demo_analysis(input_csv)

    # 3) System identification
    if identify_system:
        try:
            LOG.debug("Calling src.system_id.system_identification.identify_system")
            system_params = identify_system(analysis_result)
            analysis_result["system_parameters"] = system_params
        except Exception as exc:
            LOG.exception("System identification failed: %s. Keeping existing parameters.", exc)
    else:
        LOG.debug("No identify_system available; skipping this step (using existing params).")

    # 4) Tuning
    if tune_controller:
        try:
            LOG.debug("Calling src.optimization.tuner.tune_controller")
            tuned = tune_controller(analysis_result)
            analysis_result["tuned_gains"] = tuned
        except Exception as exc:
            LOG.exception("Tuning failed: %s. Keeping existing tuned_gains if present.", exc)
    else:
        LOG.debug("No tune_controller available; skipping tuning step (using demo or existing).")

    # 5) Save intermediate JSON (optional)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result_json_path = REPORTS_DIR / f"analysis_result_{timestamp}.json"
    if save_json:
        try:
            with open(result_json_path, "w", encoding="utf8") as fh:
                json.dump(analysis_result, fh, indent=2)
            LOG.info("Saved analysis JSON to %s", result_json_path)
            analysis_result["analysis_json"] = str(result_json_path)
        except Exception:
            LOG.exception("Failed to save analysis JSON.")

    # 6) Generate PDF
    if output_pdf_path is None:
        out_pdf = _timestamped_report_path()
    else:
        out_pdf = output_pdf_path

    if generate_pdf_report:
        try:
            LOG.debug("Calling src.utils.report_generator.generate_pdf_report")
            generate_pdf_report(analysis_result, out_pdf)
        except Exception as exc:
            LOG.exception("Custom PDF generator failed: %s. Falling back.", exc)
            out_pdf = fallback_generate_pdf(analysis_result, out_pdf)
    else:
        out_pdf = fallback_generate_pdf(analysis_result, out_pdf)

    analysis_result["report_path"] = str(out_pdf)
    LOG.info("Pipeline finished. Report saved to %s", out_pdf)
    return analysis_result


# ---------------------------
# CLI commands
# ---------------------------

@app.command()
def analyze(
    input: Optional[Path] = typer.Option(
        None, help="Path to CSV log file. If omitted and --use-demo is False, this will error."
    ),
    use_demo: bool = typer.Option(False, help="Use the bundled demo CSV from sample_data/"),
    out: Optional[Path] = typer.Option(None, help="Optional path to write the PDF report"),
) -> None:
    """
    Run the full analysis pipeline headlessly.
    """
    LOG.info("CLI: analyze called (use_demo=%s, input=%s)", use_demo, input)
    if use_demo:
        demo_csv = SAMPLE_DATA / "demo.csv"
        if not demo_csv.exists():
            LOG.error("Demo file %s not found. Place a CSV at sample_data/demo.csv", demo_csv)
            raise typer.Exit(code=2)
        chosen_csv = demo_csv
    else:
        if input is None:
            LOG.error("No input CSV provided. Use --input or --use-demo.")
            raise typer.Exit(code=2)
        chosen_csv = input
        if not chosen_csv.exists():
            LOG.error("Input file %s does not exist.", chosen_csv)
            raise typer.Exit(code=2)

    result = orchestrate_pipeline(chosen_csv, output_pdf_path=out)
    LOG.info("Analysis finished. Report: %s", result.get("report_path"))


@app.command()
def web(port: int = typer.Option(8501, help="Port for Streamlit app (if using Streamlit)")) -> None:
    """
    Launch the Streamlit web UI by invoking 'streamlit run src/webapp/app.py'.
    GitHub.dev cannot run the process; run locally when you have terminal access.
    """
    streamlit_script = SRC / "webapp" / "app.py"
    if not streamlit_script.exists():
        LOG.error("Streamlit app script not found at %s", streamlit_script)
        raise typer.Exit(code=2)

    cmd = [sys.executable, "-m", "streamlit", "run", str(streamlit_script), "--server.port", str(port)]
    LOG.info("Launching Streamlit with: %s", " ".join(cmd))
    try:
        # Spawn a new process that replaces current process (useful for direct mode)
        os.execvp(cmd[0], cmd)
    except Exception as exc:
        LOG.exception("Failed to exec Streamlit: %s", exc)
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host for uvicorn/FastAPI"),
    port: int = typer.Option(8000, help="Port for uvicorn/FastAPI"),
    app_module: str = typer.Option("src.webapp.fastapi_app:app", help="Module:app for uvicorn"),
) -> None:
    """
    Launch a FastAPI app with uvicorn. Useful if you implement FastAPI instead of Streamlit.
    Example app_module: src.webapp.fastapi_app:app
    """
    cmd = [sys.executable, "-m", "uvicorn", app_module, "--host", host, "--port", str(port), "--reload"]
    LOG.info("Running uvicorn: %s", " ".join(cmd))
    try:
        os.execvp(cmd[0], cmd)
    except Exception as exc:
        LOG.exception("Failed to exec uvicorn: %s", exc)
        raise typer.Exit(code=1)


@app.command()
def report(
    analysis_json: Optional[Path] = typer.Option(None, help="Path to saved analysis JSON to produce report from"),
    out: Optional[Path] = typer.Option(None, help="Path to write PDF output (optional)"),
) -> None:
    """
    Generate a PDF report from an existing analysis JSON (or the most recent JSON in reports/).
    """
    if analysis_json is None:
        # find the most recent analysis_result_*.json in reports
        json_files = sorted(REPORTS_DIR.glob("analysis_result_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not json_files:
            LOG.error("No analysis JSON found in %s. Run `analyze` first.", REPORTS_DIR)
            raise typer.Exit(code=2)
        analysis_json = json_files[0]
        LOG.info("Using most recent analysis JSON: %s", analysis_json)

    if not analysis_json.exists():
        LOG.error("Specified analysis JSON does not exist: %s", analysis_json)
        raise typer.Exit(code=2)

    with open(analysis_json, "r", encoding="utf8") as fh:
        analysis_result = json.load(fh)

    out_path = out if out else _timestamped_report_path(prefix="turret_report")
    if generate_pdf_report:
        try:
            generate_pdf_report(analysis_result, out_path)
            LOG.info("Report generated at %s", out_path)
        except Exception as exc:
            LOG.exception("Custom PDF generator failed: %s. Falling back.", exc)
            fallback_generate_pdf(analysis_result, out_path)
    else:
        fallback_generate_pdf(analysis_result, out_path)

    LOG.info("Report generation complete: %s", out_path)


# ---------------------------
# Entrypoint
# ---------------------------

def main() -> None:
    """
    Entrypoint for running as a module (python -m src.main).
    """
    app()


if __name__ == "__main__":
    main()

