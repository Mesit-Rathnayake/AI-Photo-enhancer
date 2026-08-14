from __future__ import annotations

import logging
from pathlib import Path
import uuid
import zipfile
import tempfile
import os

import cv2
import numpy as np
from PIL import Image

from src.models.ai_upscaler import ai_upscale, get_device_name


OUTPUT_DIRECTORY = Path("outputs")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_image(
    image: np.ndarray | None,
) -> np.ndarray:
    if image is None:
        raise ValueError("Please upload an image first.")

    if not isinstance(image, np.ndarray):
        raise TypeError("The input must be a NumPy array.")

    if image.ndim == 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB,
        )

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(
            "The image must be grayscale, RGB, or RGBA."
        )

    image = image[:, :, :3]

    if image.dtype != np.uint8:
        image = image.astype(np.float32)

        if image.max() <= 1.0:
            image *= 255.0

        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

    return image.copy()


def apply_gentle_finishing(
    image: np.ndarray,
    sharpening_strength: float = 0.0,
    saturation_adjustment: float = 0.0,
) -> np.ndarray:
    """
    Apply optional mild finishing after AI super-resolution.
    Both controls default to zero so the AI output is not unnecessarily
    altered.
    """
    result = image.copy()

    saturation_adjustment = float(
        np.clip(saturation_adjustment, -0.3, 0.3)
    )

    if saturation_adjustment != 0:
        hsv = cv2.cvtColor(
            result,
            cv2.COLOR_RGB2HSV,
        ).astype(np.float32)

        hsv[:, :, 1] *= 1.0 + saturation_adjustment
        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1],
            0,
            255,
        )

        result = cv2.cvtColor(
            hsv.astype(np.uint8),
            cv2.COLOR_HSV2RGB,
        )

    sharpening_strength = float(
        np.clip(sharpening_strength, 0.0, 0.5)
    )

    if sharpening_strength > 0:
        blurred = cv2.GaussianBlur(
            result,
            (0, 0),
            sigmaX=1.0,
        )

        result = cv2.addWeighted(
            result,
            1.0 + sharpening_strength,
            blurred,
            -sharpening_strength,
            0,
        )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


def save_image_with_ppi(
    image: np.ndarray,
    ppi: int = 300,
    jpeg_quality: int = 95,
) -> str:
    ppi = int(np.clip(ppi, 1, 1200))
    jpeg_quality = int(
        np.clip(jpeg_quality, 70, 100)
    )

    filename = (
        f"ai_enhanced_{uuid.uuid4().hex[:12]}"
        f"_{ppi}ppi.jpg"
    )

    output_path = OUTPUT_DIRECTORY / filename

    Image.fromarray(image).save(
        output_path,
        format="JPEG",
        quality=jpeg_quality,
        subsampling=0,
        optimize=True,
        dpi=(ppi, ppi),
    )

    return str(output_path.resolve())


def process_and_save(
    image: np.ndarray | None,
    ai_model: str,
    tile_size: int,
    sharpening_strength: float,
    saturation_adjustment: float,
    output_ppi: int,
    jpeg_quality: int,
    face_restoration: bool = False,
) -> tuple[np.ndarray, str, str]:
    logger.info(f"Starting single image processing with model {ai_model}, face_restoration={face_restoration}")
    original = validate_image(image)

    enhanced = ai_upscale(
        image=original,
        model_name=ai_model,
        tile_size=int(tile_size),
        face_restoration=face_restoration,
    )

    enhanced = apply_gentle_finishing(
        image=enhanced,
        sharpening_strength=sharpening_strength,
        saturation_adjustment=saturation_adjustment,
    )

    output_file = save_image_with_ppi(
        image=enhanced,
        ppi=int(output_ppi),
        jpeg_quality=int(jpeg_quality),
    )

    original_height, original_width = original.shape[:2]
    output_height, output_width = enhanced.shape[:2]

    original_mp = (
        original_width * original_height / 1_000_000
    )

    output_mp = (
        output_width * output_height / 1_000_000
    )

    print_width = output_width / int(output_ppi)
    print_height = output_height / int(output_ppi)

    device = get_device_name()

    information = f"""
### AI Processing Information

**AI model:** {ai_model}  
**Face Restoration:** {"Enabled" if face_restoration else "Disabled"}
**Processing device:** {device}  
**Tile size:** {tile_size}

**Original:** {original_width:,} × {original_height:,} pixels  
**Original resolution:** {original_mp:.2f} MP

**AI output:** {output_width:,} × {output_height:,} pixels  
**Output resolution:** {output_mp:.2f} MP

**Export metadata:** {output_ppi} PPI  
**Print size at {output_ppi} PPI:** {print_width:.2f} × {print_height:.2f} inches

The AI model may reconstruct plausible texture that was not physically
captured in the original image. Inspect faces and text carefully.
"""
    
    logger.info("Single image processing completed.")
    return enhanced, output_file, information


def process_batch(
    image_paths: list[str],
    ai_model: str,
    tile_size: int,
    sharpening_strength: float,
    saturation_adjustment: float,
    output_ppi: int,
    jpeg_quality: int,
    face_restoration: bool = False,
) -> tuple[str, str]:
    logger.info(f"Starting batch processing of {len(image_paths) if image_paths else 0} images.")
    if not image_paths:
        raise ValueError("No images uploaded for batch processing.")

    processed_files = []
    
    for idx, path in enumerate(image_paths):
        logger.info(f"Processing image {idx + 1}/{len(image_paths)}: {path}")
        try:
            pil_img = Image.open(path)
            # Convert to numpy array in RGB
            img_np = np.array(pil_img.convert("RGB"))
            
            # Process using core functions
            original = validate_image(img_np)
            enhanced = ai_upscale(
                image=original,
                model_name=ai_model,
                tile_size=int(tile_size),
                face_restoration=face_restoration,
            )
            enhanced = apply_gentle_finishing(
                image=enhanced,
                sharpening_strength=sharpening_strength,
                saturation_adjustment=saturation_adjustment,
            )
            output_file = save_image_with_ppi(
                image=enhanced,
                ppi=int(output_ppi),
                jpeg_quality=int(jpeg_quality),
            )
            processed_files.append((Path(path).name, output_file))
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")
            continue

    if not processed_files:
        raise RuntimeError("All images failed to process in the batch.")

    # Create ZIP file
    zip_filename = f"batch_enhanced_{uuid.uuid4().hex[:8]}.zip"
    zip_path = OUTPUT_DIRECTORY / zip_filename
    
    logger.info(f"Creating ZIP file: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for original_name, output_path in processed_files:
            # Prepend ai_ to original name to distinguish
            arcname = f"ai_{original_name}"
            # Ensure proper extension based on save_image_with_ppi output
            if not arcname.lower().endswith(".jpg"):
                arcname = os.path.splitext(arcname)[0] + ".jpg"
            zipf.write(output_path, arcname)

    info = f"Successfully processed {len(processed_files)} out of {len(image_paths)} images.\n"
    info += f"Model: {ai_model} | Face Restoration: {face_restoration}\n"
    info += "Batch ZIP file is ready for download."
    
    logger.info("Batch processing completed.")
    return str(zip_path.resolve()), info