from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
 
import numpy as np
import cv2
import os
from detect import detect_and_recognize_text
 
# Create app
app = FastAPI()
 
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# 1) Mount "/static" to serve files from the existing "static/" directory
app.mount("/static", StaticFiles(directory="static"), name="static")
 
# 2) Tell Jinja to look for templates (i.e. index.html) in the current folder
templates = Jinja2Templates(directory=".")
# Serve homepage
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
 
# Image OCR endpoint
@app.post("/infer/image")
async def infer_image(file: UploadFile = File(...)):
    contents = await file.read()
    img_np = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
 
    # Detect and recognize text
    results, annotated_img = detect_and_recognize_text(img)
 
    # Filter and sort results
    results = [r for r in results if r['text'].strip() and r['text'] != "[error]"]
    results.sort(key=lambda r: (round(r['bbox'][1] / 20), r['bbox'][0]))  # top-down, left-right
 
    recognized_texts = [r['text'].strip().capitalize() for r in results]
 
    save_dir = os.path.join("static")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "annotated.png")
    cv2.imwrite(save_path, annotated_img)
 
    return JSONResponse({
        "recognized_text": recognized_texts,
        "image_url": "/static/annotated.png"
    })