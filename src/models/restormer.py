"""
Restormer integration for whole-image restoration (deblurring / denoising).

This module wraps the Restormer transformer model so it can be called as a
simple function from the processing pipeline, just like `ai_upscale`.

Reference: Zamir et al., "Restormer: Efficient Transformer for High-Resolution
Image Restoration", CVPR 2022.  https://arxiv.org/abs/2111.09881
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte
from runpy import run_path


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_RESTORMER_ROOT = Path(__file__).resolve().parent.parent.parent / "external" / "Restormer"
_ARCH_FILE = _RESTORMER_ROOT / "basicsr" / "models" / "archs" / "restormer_arch.py"

TASK_CONFIGS = {
    "Motion_Deblurring": {
        "weights": _RESTORMER_ROOT / "Motion_Deblurring" / "pretrained_models" / "motion_deblurring.pth",
        "parameters": {
            "inp_channels": 3,
            "out_channels": 3,
            "dim": 48,
            "num_blocks": [4, 6, 6, 8],
            "num_refinement_blocks": 4,
            "heads": [1, 2, 4, 8],
            "ffn_expansion_factor": 2.66,
            "bias": False,
            "LayerNorm_type": "WithBias",
            "dual_pixel_task": False,
        },
    },
    "Single_Image_Defocus_Deblurring": {
        "weights": _RESTORMER_ROOT / "Defocus_Deblurring" / "pretrained_models" / "single_image_defocus_deblurring.pth",
        "parameters": {
            "inp_channels": 3,
            "out_channels": 3,
            "dim": 48,
            "num_blocks": [4, 6, 6, 8],
            "num_refinement_blocks": 4,
            "heads": [1, 2, 4, 8],
            "ffn_expansion_factor": 2.66,
            "bias": False,
            "LayerNorm_type": "WithBias",
            "dual_pixel_task": False,
        },
    },
}

# ---------------------------------------------------------------------------
# Model caching
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, torch.nn.Module] = {}
_MODEL_LOCK = Lock()


def _load_model(task: str) -> torch.nn.Module:
    """Load and cache a Restormer model for the given task."""
    with _MODEL_LOCK:
        if task in _MODEL_CACHE:
            return _MODEL_CACHE[task]

        config = TASK_CONFIGS[task]
        weights_path = config["weights"]

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Restormer weights not found at {weights_path}. "
                "Please download the pretrained model."
            )

        # Dynamically load the Restormer architecture
        load_arch = run_path(str(_ARCH_FILE))
        model = load_arch["Restormer"](**config["parameters"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        checkpoint = torch.load(str(weights_path), map_location=device)
        model.load_state_dict(checkpoint["params"])
        model.eval()

        _MODEL_CACHE[task] = model
        logger.info(f"Restormer model loaded for task: {task}")
        return model


def restore_image(
    image: np.ndarray,
    task: str = "Motion_Deblurring",
    tile_size: int | None = 512,
    tile_overlap: int = 32,
) -> np.ndarray:
    """
    Restore an RGB uint8 NumPy image using Restormer.

    Parameters
    ----------
    image : np.ndarray
        Input image in RGB uint8 format (H, W, 3).
    task : str
        One of the keys in TASK_CONFIGS.
    tile_size : int or None
        Process the image in tiles of this size. None = full resolution
        (uses more memory). Must be a multiple of 8.
    tile_overlap : int
        Overlap between tiles.

    Returns
    -------
    np.ndarray
        Restored image in RGB uint8 format (H, W, 3).
    """
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unknown restoration task: {task}. Choose from: {list(TASK_CONFIGS.keys())}")

    logger.info(f"Running Restormer restoration: task={task}, tile_size={tile_size}")

    model = _load_model(task)
    device = next(model.parameters()).device

    img_multiple_of = 8

    # Convert to float tensor
    input_tensor = (
        torch.from_numpy(image)
        .float()
        .div(255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    _, _, height, width = input_tensor.shape

    # Pad to multiple of 8
    padh = (img_multiple_of - height % img_multiple_of) % img_multiple_of
    padw = (img_multiple_of - width % img_multiple_of) % img_multiple_of
    input_tensor = F.pad(input_tensor, (0, padw, 0, padh), "reflect")

    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        if tile_size is None:
            restored = model(input_tensor)
        else:
            # Tiled inference to avoid OOM
            tile = tile_size
            if tile % 8 != 0:
                tile = (tile // 8) * 8
            b, c, h, w = input_tensor.shape
            tile = min(tile, h, w)
            stride = tile - tile_overlap

            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = torch.zeros(b, c, h, w).type_as(input_tensor)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_tensor[..., h_idx : h_idx + tile, w_idx : w_idx + tile]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx : h_idx + tile, w_idx : w_idx + tile].add_(out_patch)
                    W[..., h_idx : h_idx + tile, w_idx : w_idx + tile].add_(out_patch_mask)

            restored = E.div_(W)

    restored = torch.clamp(restored, 0, 1)
    # Unpad
    restored = restored[:, :, :height, :width]
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])

    logger.info("Restormer restoration complete.")
    return restored
