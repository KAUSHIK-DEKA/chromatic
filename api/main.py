"""
FastAPI backend for Color Extractor.

Endpoints:
  GET  /          -> serve the frontend (index.html)
  POST /extract   -> accept zip upload + mode, return xlsx file

Output columns: SKU, Color, Color Code, Parent Color
"""
import io
import os
import zipfile
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

from color_extractor import process_image

app = FastAPI(title="Garment Color Extractor")

STATIC_DIR = Path(__file__).parent.parent / "static"

IMAGE_EXTS = {".webp", ".jpg", ".jpeg", ".png"}


@app.get("/")
def serve_frontend():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    raise HTTPException(status_code=404, detail="index.html not found")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract_colors(
    file: UploadFile = File(...),
    mode: Literal["garment", "footwear"] = Form("garment"),
):
    """Accept a zip of images, return xlsx with SKU, Color, Color Code, Parent Color."""
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file")

    zip_bytes = await file.read()
    if len(zip_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    rows = []
    errors = []

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            entries = []
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                basename = os.path.basename(name)
                if not basename or basename.startswith("."):
                    continue
                if "__MACOSX" in name:
                    continue
                ext = os.path.splitext(basename)[1].lower()
                if ext in IMAGE_EXTS:
                    entries.append((basename, name))

            if not entries:
                raise HTTPException(
                    status_code=400,
                    detail="No image files (.webp/.jpg/.jpeg/.png) found in the zip",
                )

            entries.sort(key=lambda x: x[0])

            for basename, zip_path in entries:
                sku = os.path.splitext(basename)[0]
                try:
                    with zf.open(zip_path) as img_file:
                        img_bytes = io.BytesIO(img_file.read())
                    result = process_image(img_bytes, mode=mode)
                    rows.append((sku, result["name"], result["hex"], result["parent"]))
                except Exception as e:
                    rows.append((sku, "ERROR", "", str(e)[:60]))
                    errors.append(f"{sku}: {e}")
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid or corrupted zip file")

    # Build xlsx with new column schema
    wb = Workbook()
    ws = wb.active
    ws.title = mode.capitalize()

    headers = ["SKU", "Color", "Color Code", "Parent Color"]
    header_font = Font(name="Arial", bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill("solid", start_color="2F5597")
    header_align = Alignment(horizontal="center", vertical="center")
    body_font = Font(name="Arial", size=11)
    body_align = Alignment(horizontal="left", vertical="center")
    thin = Side(border_style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for col, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        cell.border = border

    for i, (sku, color, code, parent) in enumerate(rows, start=2):
        for col, val in enumerate([sku, color, code, parent], start=1):
            cell = ws.cell(row=i, column=col, value=val)
            cell.font = body_font
            cell.alignment = body_align
            cell.border = border

    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 26
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 16
    ws.freeze_panes = "A2"

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    source_name = os.path.splitext(file.filename)[0]
    out_name = f"{source_name}_colors.xlsx"

    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{out_name}"',
            "X-Processed-Count": str(len(rows)),
            "X-Error-Count": str(len(errors)),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
