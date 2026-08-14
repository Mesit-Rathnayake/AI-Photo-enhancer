# 🎨 AI Photo Enhancer Pro

A production-ready, full-stack application that leverages advanced AI models for premium image super-resolution, face restoration, and classical image enhancement.

Built with a lightning-fast **React/Vite** frontend and a high-performance **FastAPI** backend, this tool transforms low-quality images into stunning, high-resolution masterpieces.

## ✨ Features

- **AI Super-Resolution**: Upscale images by 2x or 4x without losing quality.
  - *RealESRGAN x2 (Balanced)*
  - *RealESRGAN x4 (Max Detail)*
  - *RealESRGAN Anime x4 (Optimized for illustrations and 2D art)*
- **AI Face Restoration**: Seamless integration with **GFPGAN** to automatically detect and restore heavily degraded faces during the upscaling process.
- **Batch Processing**: Upload entire folders of images and download all enhanced results in a single ZIP file.
- **Classical Enhancements**: Fine-tune your results with post-processing adjustments:
  - Sharpening strength
  - Saturation adjustments
  - Custom Output PPI (Web 72 PPI vs Print 300/600 PPI)
  - JPEG Quality compression control
- **Smart Model Management**: AI weights are massive. This app keeps the repository clean by automatically downloading the necessary `.pth` models on the first run.
- **Modern UI**: A bespoke, responsive Dark Mode glassmorphism interface with an interactive before/after image comparison slider.

## 🏗️ Architecture

- **Frontend**: React, TypeScript, Vite, CSS (Glassmorphism design system)
- **Backend**: Python, FastAPI, Uvicorn
- **AI Processing**: PyTorch, RealESRGAN, GFPGAN, OpenCV, Pillow

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+ (for frontend)
- (Optional but highly recommended) NVIDIA GPU with CUDA for fast AI processing.

### 1. Setup the Backend

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/Mesit-Rathnayake/AI-Photo-enhancer.git
cd AI-Photo-enhancer

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install the backend dependencies
pip install -r requirements.txt
```

### 2. Setup the Frontend

Open a second terminal window, navigate to the `frontend` folder, and install the Node modules:

```bash
cd frontend
npm install
```

### 3. Run the Application

Start both the backend server and the frontend development server:

**Terminal 1 (Backend):**
```bash
uvicorn api:app --reload
```

**Terminal 2 (Frontend):**
```bash
npm run dev
```

Open your browser and navigate to `http://localhost:5173`. 
*(Note: The AI models will automatically download the very first time you process an image. This may take a few minutes depending on your internet connection).*

## 🛠️ Configuration

You can adjust the **Tile Size** in the UI to manage GPU VRAM consumption:
- `128`: Low VRAM (slower)
- `256`: Balanced (default)
- `512`: High VRAM (fastest)

## 📝 License
This project is open-source. Please check the respective licenses for the RealESRGAN and GFPGAN models.
