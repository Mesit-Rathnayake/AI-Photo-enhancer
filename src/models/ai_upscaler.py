from __future__ import annotations

from pathlib import Path
from threading import Lock
import urllib.request

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
try:
    from gfpgan import GFPGANer
    HAS_GFPGAN = True
except ImportError:
    HAS_GFPGAN = False


MODEL_DIRECTORY = Path("models")
MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "RealESRGAN x2": {
        "filename": "RealESRGAN_x2plus.pth",
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/releases/"
            "download/v0.2.1/RealESRGAN_x2plus.pth"
        ),
        "scale": 2,
        "num_block": 23,
    },
    "RealESRGAN x4": {
        "filename": "RealESRGAN_x4plus.pth",
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/releases/"
            "download/v0.1.0/RealESRGAN_x4plus.pth"
        ),
        "scale": 4,
        "num_block": 23,
    },
    "RealESRGAN Anime x4": {
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/releases/"
            "download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        ),
        "scale": 4,
        "num_block": 6,
    },
}


_UPSCALER_CACHE: dict[str, RealESRGANer] = {}
_MODEL_LOCK = Lock()


def get_device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)

    return "CPU"


def download_model(
    model_name: str,
) -> Path:
    """
    Download the selected pretrained Real-ESRGAN model when it is not
    available locally.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported AI model: {model_name}")

    config = MODEL_CONFIGS[model_name]

    model_path = MODEL_DIRECTORY / config["filename"]

    if model_path.exists():
        return model_path

    try:
        urllib.request.urlretrieve(
            config["url"],
            model_path,
        )
    except Exception as error:
        if model_path.exists():
            model_path.unlink()

        raise RuntimeError(
            "The Real-ESRGAN model could not be downloaded. "
            "Check your internet connection and try again."
        ) from error

    return model_path


def create_upscaler(
    model_name: str,
    tile_size: int = 256,
) -> RealESRGANer:
    """
    Create and cache a Real-ESRGAN inference object.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported AI model: {model_name}")

    cache_key = f"{model_name}_{tile_size}"

    with _MODEL_LOCK:
        if cache_key in _UPSCALER_CACHE:
            return _UPSCALER_CACHE[cache_key]

        config = MODEL_CONFIGS[model_name]
        model_path = download_model(model_name)

        scale = int(config["scale"])

        network = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=int(config["num_block"]),
            num_grow_ch=32,
            scale=scale,
        )

        upscaler = RealESRGANer(
            scale=scale,
            model_path=str(model_path),
            model=network,
            tile=max(0, int(tile_size)),
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
            gpu_id=0 if torch.cuda.is_available() else None,
        )

        _UPSCALER_CACHE[cache_key] = upscaler

        return upscaler


def get_face_enhancer(
    upscaler: RealESRGANer | None = None,
) -> GFPGANer | None:
    if not HAS_GFPGAN:
        return None
    
    # Download GFPGAN model if needed
    model_name = "GFPGANv1.4.pth"
    model_path = MODEL_DIRECTORY / model_name
    if not model_path.exists():
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
        try:
            urllib.request.urlretrieve(url, model_path)
        except Exception as error:
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError("Failed to download GFPGAN model.") from error

    face_enhancer = GFPGANer(
        model_path=str(model_path),
        upscale=upscaler.scale if upscaler is not None else 1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upscaler,
    )
    return face_enhancer


def soften_face_wrinkles(
    image: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Apply gentle smoothing inside detected face regions only."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0:
        return image

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(24, 24),
    )
    result = image.copy()

    for x, y, width, height in faces:
        smoothed = cv2.bilateralFilter(
            result[y : y + height, x : x + width],
            d=7,
            sigmaColor=35,
            sigmaSpace=35,
        )
        mask = np.zeros((height, width), dtype=np.float32)
        center = (width // 2, int(height * 0.52))
        axes = (max(1, int(width * 0.43)), max(1, int(height * 0.43)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, width / 14))
        alpha = (mask * strength)[:, :, None]
        region = result[y : y + height, x : x + width].astype(np.float32)
        result[y : y + height, x : x + width] = (
            smoothed.astype(np.float32) * alpha + region * (1.0 - alpha)
        ).astype(np.uint8)

    return result


def preserve_original_colors(
    restored: np.ndarray,
    original: np.ndarray,
) -> np.ndarray:
    """Keep reconstructed luminance while restoring the source photo palette."""
    source = cv2.resize(
        original,
        (restored.shape[1], restored.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    restored_lab = cv2.cvtColor(restored, cv2.COLOR_RGB2LAB)
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
    restored_lab[:, :, 1:] = source_lab[:, :, 1:]
    return cv2.cvtColor(restored_lab, cv2.COLOR_LAB2RGB)


def ai_upscale(
    image: np.ndarray,
    model_name: str = "RealESRGAN x2",
    tile_size: int = 256,
    face_restoration: bool = False,
    skin_smoothing: float = 0.0,
    preserve_colors: bool = True,
) -> np.ndarray:
    """
    Upscale an RGB NumPy image using Real-ESRGAN, with optional GFPGAN face restoration.
    If model_name is None / Skip, retains native resolution while supporting optional face restoration.
    """
    if image is None:
        raise ValueError("No image was supplied for AI upscaling.")

    if not isinstance(image, np.ndarray):
        raise TypeError("The image must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("The AI upscaler requires an RGB image.")

    is_skip_upscale = model_name in ("None", "None (Skip)", "none", "Skip")

    if is_skip_upscale:
        if not face_restoration:
            return image.copy()
        face_enhancer = get_face_enhancer(None)
        if face_enhancer is None:
            raise RuntimeError("Face restoration requested but GFPGAN could not be loaded.")
        
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, _, output_bgr = face_enhancer.enhance(
            bgr_image,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.75,
        )
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        if preserve_colors:
            output_rgb = preserve_original_colors(output_rgb, image)
        output_rgb = soften_face_wrinkles(output_rgb, skin_smoothing)
        return output_rgb

    upscaler = create_upscaler(
        model_name=model_name,
        tile_size=tile_size,
    )

    face_enhancer = None
    if face_restoration:
        face_enhancer = get_face_enhancer(upscaler)
        if face_enhancer is None:
            raise RuntimeError("Face restoration requested but GFPGAN could not be loaded.")

    bgr_image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    try:
        if face_enhancer is not None:
            _, _, output_bgr = face_enhancer.enhance(
                bgr_image,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.75,
            )
        else:
            output_bgr, _ = upscaler.enhance(
                bgr_image,
                outscale=MODEL_CONFIGS[model_name]["scale"],
            )
    except RuntimeError as error:
        message = str(error).lower()

        if "out of memory" in message:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise RuntimeError(
                "The GPU ran out of memory. Reduce the AI tile size "
                "to 128, or use the 2× model."
            ) from error

        raise
    except MemoryError as error:
        raise RuntimeError(
            "System ran out of RAM while processing this massive image. "
            "Please use a smaller input photo, disable Face Restoration, or use the 2x model."
        ) from error

    output_rgb = cv2.cvtColor(
        output_bgr,
        cv2.COLOR_BGR2RGB,
    )

    if face_enhancer is not None:
        if preserve_colors:
            output_rgb = preserve_original_colors(output_rgb, image)
        output_rgb = soften_face_wrinkles(output_rgb, skin_smoothing)

    return output_rgb