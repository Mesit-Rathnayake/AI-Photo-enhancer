# AI Photo Enhancer

![AI Photo Enhancer](frontend/src/assets/hero.png)

**Restore the detail. Keep the memory.**

AI Photo Enhancer is a local-first restoration workspace for bringing new clarity to old portraits, scanned photographs, and low-quality images without forcing them into an artificial modern look.

It combines a FastAPI backend with a Vite + React frontend and a carefully constrained Real-ESRGAN, Restormer, and GFPGAN pipeline. Your images stay on your machine while you improve resolution, recover facial detail, and export print-ready results.

## Highlights

- AI upscaling with Real-ESRGAN models
- Face restoration using GFPGAN
- Deblurring and restoration using Restormer
- Old-photo cleanup mode with color-preserving enhancement
- Optional skin smoothing for face cleanup
- Optional preservation of original colors during face reconstruction
- Batch processing for multiple images
- JPEG export with configurable PPI and quality settings

## Product launch

AI Photo Enhancer is built for people restoring family archives, scanned albums, and portrait collections. Its guiding principle is simple: recover believable detail while preserving the character, colors, and texture of the original photograph.

The current release focuses on portraits and face restoration. It does not attempt aggressive full-body or clothing reconstruction, because constrained enhancement produces more natural results for the image collection this project targets.

## Privacy by design

The application runs locally. Images are uploaded only to the local FastAPI service running on your computer, processed there, and written to the local `outputs/` directory. Model weights, generated files, virtual environments, caches, and local test fixtures are excluded from version control.

## Current workflow

The app is intentionally optimized for a face-focused restoration pipeline rather than aggressive full-body reconstruction. In practice, the best results come from:

- enabling face restoration for portraits
- using mild skin smoothing to reduce wrinkle emphasis
- preserving original colors to avoid unnatural face tinting
- keeping the rest of the image natural instead of over-restoring clothing or body details

This makes the tool especially effective for older portrait photos where the face should look restored without becoming plastic or over-processed.

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
├── models/                   # downloaded model weights
├── gfpgan/weights/           # GFPGAN weights
├── outputs/                  # generated enhanced images and ZIP exports
├── requirements.txt
├── README.md
└── .venv/                    # local Python environment
```

## Features in detail

### AI enlargement

Choose from:

- RealESRGAN x2
- RealESRGAN x4
- RealESRGAN Anime x4

These are used to increase image resolution while trying to preserve detail and natural texture.

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

## Local setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- Windows, macOS, or Linux
- Optional: NVIDIA GPU for faster processing

### 1. Clone and create the environment

```bash
git clone https://github.com/Mesit-Rathnayake/AI-Photo-enhancer.git
cd AI-Photo-enhancer

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Start the backend

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

### 5. Start the frontend

In a second terminal:

```bash
cd frontend
npm run dev
```

Then open:

- Frontend: http://localhost:5173
- API docs: http://localhost:8000/docs

## API endpoints

The backend exposes two multipart upload endpoints:

- `POST /api/enhance/single` for one image
- `POST /api/enhance/batch` for multiple images and a ZIP result

Interactive OpenAPI documentation is available at `http://localhost:8000/docs` while the backend is running.

## Model notes

This project relies on pretrained weights for the enhancement models. The repository includes prebuilt model assets in the following locations when present:

- `models/`
- `gfpgan/weights/`

If a required weight file is missing, the app may attempt to fetch or rely on a local model path during processing. The exact behavior depends on the model loader and your local environment.

## Common usage tips

- Start with `RealESRGAN x2` for a safer default.
- Enable face restoration for portrait images.
- Keep skin smoothing modest unless you want a more polished face finish.
- Keep color preservation on for old portraits and archival images.
- Use older photo mode for scanned, faded, or paper-texture images.

## Development notes

This app is intended for local running and experimentation. It is not a cloud-hosted SaaS product. The code is structured so it can be tested and tuned on a workstation GPU or CPU, depending on the available hardware.

Before opening a public pull request:

- run `python -m compileall api.py app.py src`
- run `npm run build` from `frontend/`
- confirm no model weights, generated outputs, personal images, or local environment files are staged
- review third-party model licenses before distributing builds

## License

Please check the licensing for the third-party AI models and dependencies used in this project before redistribution or commercial deployment.
