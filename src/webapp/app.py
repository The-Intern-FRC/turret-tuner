from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import pandas as pd
from analysis.analyzer import run_bayesian_analysis
from optimization.tuner import suggest_optimal_coefficients
from system_id.system_identification import identify_system
from utils.data_loader import load_sample_data
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import json

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR.parent.parent / "reports"

UPLOAD_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Turret Tuner Dashboard", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# -----------------------------------------------------------------------------
# Helper: Generate PDF report
# -----------------------------------------------------------------------------
def generate_pdf_report(analysis_results: str, tuned_values: dict, system_info: str, filename: str):
    report_path = REPORTS_DIR / filename
    c = canvas.Canvas(str(report_path), pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Turret Tuner: Bayesian Analysis Report")

    # Suggested coefficients
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, "Suggested Coefficients:")
    y = height - 120
    for k, v in tuned_values.items():
        c.drawString(70, y, f"{k}: {v}")
        y -= 20

    # System info
    c.drawString(50, y - 10, f"System ID: {system_info}")

    # Analysis summary
    c.drawString(50, y - 40, "Analysis Summary:")
    text_object = c.beginText(50, y - 60)
    for line in analysis_results.split("\n"):
        text_object.textLine(line)
    c.drawText(text_object)

    c.showPage()
    c.save()
    return report_path

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/run_test", response_class=JSONResponse)
async def run_test_data():
    df = load_sample_data()
    analysis_results = run_bayesian_analysis(df)
    tuned_values = suggest_optimal_coefficients(analysis_results)
    system_info = identify_system(df)

    pdf_path = generate_pdf_report(str(analysis_results), tuned_values, system_info, "demo_report.pdf")

    return JSONResponse({
        "tuning_values": tuned_values,
        "pdf_url": f"/download/{pdf_path.name}"
    })

@app.post("/upload", response_class=JSONResponse)
async def upload_logs(file: UploadFile):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df = pd.read_csv(file_path)
    analysis_results = run_bayesian_analysis(df)
    tuned_values = suggest_optimal_coefficients(analysis_results)
    system_info = identify_system(df)

    pdf_filename = f"{file.filename.rsplit('.',1)[0]}_report.pdf"
    pdf_path = generate_pdf_report(str(analysis_results), tuned_values, system_info, pdf_filename)

    return JSONResponse({
        "tuning_values": tuned_values,
        "pdf_url": f"/download/{pdf_path.name}"
    })

@app.get("/download/{filename}")
async def download_report(filename: str):
    file_path = REPORTS_DIR / filename
    if not file_path.exists():
        return {"error": "Report not found"}
    return FileResponse(path=file_path, filename=filename, media_type="application/pdf")

# -----------------------------------------------------------------------------
# Run (local only)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("Starting Turret Tuner web dashboard at http://127.0.0.1:8000 ...")
    uvicorn.run("webapp.app:app", host="127.0.0.1", port=8000, reload=True)
