from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import cv2
import numpy as np
from typing import Optional
import json

# Import pipeline
from backend import edge_pipeline

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/process")
async def process_image(
    file: UploadFile = File(...),
    scale: int = Form(4),
    edge_width_hr: float = Form(1.5),
    auto_canny: bool = Form(True),
    canny_low: int = Form(80),
    canny_high: int = Form(160),
    blur_sigma: float = Form(1.0),
    thinning: bool = Form(True),
    methods: str = Form("baseline,sdf,pde")
):
    # Read image
    contents = await file.read()
    img_bgr = edge_pipeline.load_image(contents)
    
    if img_bgr is None:
        return {"error": "Invalid image"}
        
    h, w = img_bgr.shape[:2]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Compute Edge Map (LR)
    # Returns 0/1 binary map
    edge_lr_01 = edge_pipeline.compute_edge_map(
        img_gray, 
        auto_canny=auto_canny,
        canny_low=canny_low,
        canny_high=canny_high,
        blur_sigma=blur_sigma,
        thinning=thinning
    )
    
    # Prepare results dict
    results = {}
    
    # Helper to encode
    def encode(img):
        return edge_pipeline.encode_png_base64(img)
    
    # Original Image (Base64)
    results["original"] = encode(img_bgr)
    
    # LR Edge Map (Visualize as 0/255)
    results["edge_lr"] = encode(edge_lr_01 * 255)
    
    method_list = [m.strip() for m in methods.split(",")]
    
    # Run Upscaling Methods
    if "baseline" in method_list:
        res_baseline = edge_pipeline.upscale_baseline(edge_lr_01, scale, thinning=thinning)
        results["baseline"] = encode(res_baseline)
        
    if "sdf" in method_list:
        res_sdf = edge_pipeline.upscale_sdf(edge_lr_01, scale, edge_width_hr, thinning=thinning)
        results["sdf"] = encode(res_sdf)
        
    if "pde" in method_list:
        # Default PDE params as per spec
        res_pde = edge_pipeline.upscale_pde(
            edge_lr_01, 
            scale, 
            iters=20, 
            dt=0.2, 
            lambda_fid=1.0, 
            eps=1e-6, 
            thinning=thinning
        )
        results["pde"] = encode(res_pde)

    return {
        "meta": {
            "scale": scale,
            "input_size": [h, w],
            "edge_size": [h, w],
            "output_size": [h * scale, w * scale]
        },
        "images": results
    }

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
