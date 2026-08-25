# AI Photo Enhancer

![AI Photo Enhancer](frontend/src/assets/hero.png)

**Restore the detail. Keep the memory.**

AI Photo Enhancer is a portfolio project exploring how modern computer-vision models can be composed into a practical, human-centered image restoration product. It brings clarity to old portraits, scanned photographs, and low-quality images without forcing them into an artificial modern look.

The project demonstrates full-stack product thinking: a React interface, a typed API boundary, an image-processing orchestration layer, model adapters, classical computer-vision finishing, and local-first privacy. The result is a focused restoration workflow that is explainable, tunable, and usable on a workstation.

## Highlights

- AI upscaling with Real-ESRGAN models
- Face restoration using GFPGAN
- Deblurring and restoration using Restormer
- Old-photo cleanup mode with color-preserving enhancement
- Optional skin smoothing for face cleanup
- Optional preservation of original colors during face reconstruction
- Digital 2x quality recovery for noise and softness from phone zoom
- Conservative 1x lens distortion correction with source dimensions preserved
- Stronger wide-angle portrait correction for fisheye and edge stretching
- Radial lens-distortion correction that preserves subject proportions
- Batch processing for multiple images
- JPEG export with configurable PPI and quality settings

## Portfolio overview

This project was designed as a complete product rather than a model demo. It turns several specialized restoration systems into one coherent workflow for family archives, scanned albums, and portrait collections.

The guiding engineering decision is restraint: recover believable detail while preserving the character, colors, and texture of the source. The current release focuses on portraits and face restoration instead of aggressive full-body or clothing reconstruction, because constrained enhancement produces more natural results for this image family.

### Skills demonstrated

- Full-stack application architecture with React, FastAPI, and Python
- AI model integration across Real-ESRGAN, Restormer, and GFPGAN
- Image-processing pipelines with OpenCV, NumPy, and Pillow
- Multipart upload and batch ZIP workflows
- Resource-aware inference using configurable tile sizes
- Face-region post-processing and color-preservation strategies
- Responsive interface design with before-and-after visual comparison
- Defensive validation, error handling, logging, and export metadata

## Privacy by design

The application runs locally. Images are uploaded only to the local FastAPI service running on your computer, processed there, and written to the local `outputs/` directory. Model weights, generated files, virtual environments, caches, and local test fixtures are excluded from version control.

## Architecture

The application is organized into four layers:

1. **Presentation layer**: React and TypeScript manage uploads, settings, processing state, errors, and the before/after comparison view.
2. **API layer**: FastAPI exposes single-image and batch endpoints, validates multipart inputs, and returns JPEG or ZIP responses.
3. **Processing layer**: the orchestration service coordinates optional old-photo cleanup, Restormer deblurring, AI upscaling, face restoration, finishing, and export.
4. **Model layer**: dedicated adapters isolate Real-ESRGAN, GFPGAN, and Restormer loading and inference details from the application workflow.

```mermaid
flowchart LR
	A[React + TypeScript UI] --> B[FastAPI upload endpoints]
	B --> C[Image validation]
	C --> D{Old photo mode}
	D -->|enabled| E[Classical denoise + sharpen]
	D -->|disabled| F[Restormer deblur]
	E --> F
	F --> G[Real-ESRGAN upscale]
	G --> H{Face restoration}
	H -->|enabled| I[GFPGAN + skin smoothing + color blend]
	H -->|disabled| J[Preserve AI output]
	I --> K[Finishing + PPI/JPEG export]
	J --> K
	K --> L[JPEG or batch ZIP response]
```

### Processing pipeline

The final enhancement pass applies face restoration only when requested. Optional intermediate passes upscale and downscale the image to reinforce textures without repeatedly altering facial identity. Restormer is used as a task-specific restoration stage, while classical OpenCV processing handles gentle old-photo cleanup.

Face processing is deliberately localized. GFPGAN reconstructs facial structure, skin smoothing softens excessive wrinkle emphasis inside detected faces, and color preservation blends the result toward the original palette. This keeps clothing, background texture, and the overall photographic character natural.

## Tech stack

- Backend: Python, FastAPI, Uvicorn
- Frontend: React, TypeScript, Vite
- AI tools: Real-ESRGAN, GFPGAN, Restormer, OpenCV, PyTorch, Pillow

## Project structure

```text
Hybrid-Photo-Enhancer/
├── api.py                     # FastAPI endpoints
├── app.py                    # Gradio app entry point
├── frontend/                 # React frontend
├── src/
│   ├── models/
│   │   ├── ai_upscaler.py
│   │   └── restormer.py
│   └── processing/
│       └── classical_enhancer.py
├── requirements.txt          # Python dependencies
├── README.md                 # Portfolio and architecture documentation
└── outputs/                  # generated files, excluded from Git
```

Model weights, generated outputs, environments, caches, and personal test images are intentionally excluded from the public repository.

## Features in detail

### AI enlargement

Choose from:

- RealESRGAN x2
- RealESRGAN x4
- RealESRGAN Anime x4

These are used to increase image resolution while trying to preserve detail and natural texture.

### Camera-aware correction

The workflow addresses two common limitations of mid-range phone cameras:

- **Digital 2x quality recovery:** applies restrained denoising and detail recovery before AI upscaling to reduce the softness and noise introduced when a phone crops into its 1x sensor instead of using optical zoom.
- **1x lens correction:** applies a mild barrel-distortion correction to wide-angle 1x captures while keeping the original pixel dimensions and avoiding additional sharpening or enlargement.
- **Wide-angle portrait correction:** applies a stronger radial correction for close portraits with fisheye-like faces or an exaggerated foreground body. It preserves output dimensions, but the best result still depends on how centered the subject is and how much of the frame is available for correction.
- **Radial lens-distortion correction:** applies a global radial correction based on distance from the optical center. It corrects bent lines and edge stretching without applying body-slimming or vertical compression.

These are separate capture modes so the user can choose whether the input needs quality recovery or geometry correction, rather than applying both transformations blindly.

### Face restoration

When enabled, the app applies GFPGAN face enhancement to improve facial structure and detail. This works best on portraits and degraded face crops.

### Skin smoothing

This control softly reduces fine wrinkles in detected facial regions without over-smoothing the full image. It was added to prevent the face from becoming too harsh or over-wrinkled after enhancement.

### Preserve original colors

This keeps the face closer to the original photo palette while still improving detail and structure. It is recommended for old photo restoration and portrait work.

### Old photo mode

Old-photo mode applies gentle classical denoising and sharpening first, then continues with the AI restoration pipeline. It is useful for worn, faded, or noisy historical images.

### Batch enhancement

Multiple images can be uploaded at once and exported as a ZIP archive of enhanced JPEG outputs.

### Export controls

The app supports:

- sharpening strength
- saturation adjustment
- output PPI
- JPEG quality
- GPU tile size selection

## API endpoints

The backend exposes two multipart upload endpoints:

- `POST /api/enhance/single` for one image
- `POST /api/enhance/batch` for multiple images and a ZIP result

The API boundary keeps transport concerns separate from processing concerns. Interactive OpenAPI documentation is available at `/docs` when the local service is running.

## Model notes

This project relies on pretrained weights for the enhancement models. The repository includes prebuilt model assets in the following locations when present:

- `models/`
- `gfpgan/weights/`

If a required weight file is missing, the app may attempt to fetch or rely on a local model path during processing. The exact behavior depends on the model loader and your local environment.

## Engineering considerations

- **Natural results over maximum alteration:** face restoration is opt-in and localized.
- **Hardware flexibility:** tile size controls memory use, making inference more practical across different GPUs.
- **Camera-aware processing:** digital zoom quality recovery and 1x lens correction are explicit, opt-in stages with different goals.
- **Honest correction boundaries:** lens correction can reduce optical and perspective exaggeration, but it cannot recreate a telephoto viewpoint or fully undo a subject photographed too close with a wide lens.
- **Proportion-safe correction:** radial correction leaves the subject proportions untouched; it cannot recreate the viewpoint of a longer lens or undo perspective caused by standing too close to a wide lens.
- **Failure containment:** image validation, HTTP errors, logging, and batch-level exception handling keep one failed image from hiding the result of an entire batch.
- **Output fidelity:** PPI, JPEG quality, dimensions, and print-size metadata are preserved in the export workflow.
- **Privacy:** images are processed by a local service and are not sent to a third-party hosted API.

## Validation

The project has been checked with:

- run `python -m compileall api.py app.py src`
- run `npm run build` from `frontend/`
- run `npm run lint` from `frontend/`
- confirm no model weights, generated outputs, personal images, or local environment files are staged

## License

Please check the licensing for the third-party AI models and dependencies used in this project before redistribution or commercial deployment.
