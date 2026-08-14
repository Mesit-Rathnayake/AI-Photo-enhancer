import os
from pathlib import Path
import tempfile
import uuid
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

from src.processing.classical_enhancer import process_and_save, process_batch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Photo Enhancer API")

# Allow Vite frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/enhance/single")
async def enhance_single(
    image: UploadFile = File(...),
    ai_model: str = Form("RealESRGAN x2"),
    tile_size: int = Form(256),
    sharpening_strength: float = Form(0.0),
    saturation_adjustment: float = Form(0.0),
    output_ppi: int = Form(300),
    jpeg_quality: int = Form(95),
    face_restoration: bool = Form(False),
):
    try:
        # Read image to numpy array
        contents = await image.read()
        
        # Save temp to read with PIL
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
            
        pil_img = Image.open(tmp_path).convert("RGB")
        img_np = np.array(pil_img)
        
        # Clean up temp input
        os.unlink(tmp_path)
        
        enhanced_np, output_file, info = process_and_save(
            image=img_np,
            ai_model=ai_model,
            tile_size=tile_size,
            sharpening_strength=sharpening_strength,
            saturation_adjustment=saturation_adjustment,
            output_ppi=output_ppi,
            jpeg_quality=jpeg_quality,
            face_restoration=face_restoration,
        )
        
        # Verify the file was created
        if not os.path.exists(output_file):
            raise HTTPException(status_code=500, detail="Enhanced image not generated.")
            
        return FileResponse(
            output_file, 
            media_type="image/jpeg", 
            filename=Path(output_file).name,
        )

    except Exception as e:
        logger.error(f"Error processing single image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/enhance/batch")
async def enhance_batch(
    images: list[UploadFile] = File(...),
    ai_model: str = Form("RealESRGAN x2"),
    tile_size: int = Form(256),
    sharpening_strength: float = Form(0.0),
    saturation_adjustment: float = Form(0.0),
    output_ppi: int = Form(300),
    jpeg_quality: int = Form(95),
    face_restoration: bool = Form(False),
):
    if not images:
        raise HTTPException(status_code=400, detail="No images provided.")

    temp_paths = []
    try:
        # Save uploaded files temporarily
        for file in images:
            contents = await file.read()
            # Need a valid extension for PIL to guess format
            ext = Path(file.filename).suffix if file.filename else ".jpg"
            if not ext: ext = ".jpg"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(contents)
                temp_paths.append(tmp.name)

        # Process the batch
        zip_path, info = process_batch(
            image_paths=temp_paths,
            ai_model=ai_model,
            tile_size=tile_size,
            sharpening_strength=sharpening_strength,
            saturation_adjustment=saturation_adjustment,
            output_ppi=output_ppi,
            jpeg_quality=jpeg_quality,
            face_restoration=face_restoration,
        )
        
        if not os.path.exists(zip_path):
            raise HTTPException(status_code=500, detail="Batch ZIP not generated.")
            
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=Path(zip_path).name,
        )
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary input files
        for tmp_path in temp_paths:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
