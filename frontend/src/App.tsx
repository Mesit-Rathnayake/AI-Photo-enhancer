import { useRef, useState } from 'react';
import {
  AlertTriangle,
  Download,
  FolderUp,
  ImageIcon,
  Layers,
  Sliders,
  Sparkles,
  Upload,
  X,
  Zap,
} from 'lucide-react';
import './App.css';

const API_URL = 'http://127.0.0.1:8000';

type SettingsTab = 'engine' | 'restoration' | 'finishing';

function App() {
  const [tab, setTab] = useState<'single' | 'batch'>('single');
  const [file, setFile] = useState<File | null>(null);
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [sliderPos, setSliderPos] = useState(50);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);

  // Engine Settings
  const [model, setModel] = useState('RealESRGAN x2');
  const [captureMode, setCaptureMode] = useState('Standard');
  const [passes, setPasses] = useState('1');
  const [tileSize, setTileSize] = useState('256');

  // Restoration Settings
  const [restorationTask, setRestorationTask] = useState('None');
  const [faceRestoration, setFaceRestoration] = useState(false);
  const [skinSmoothing, setSkinSmoothing] = useState('0.2');
  const [preserveColors, setPreserveColors] = useState(true);
  const [oldPhotoMode, setOldPhotoMode] = useState(false);

  // Finishing Settings
  const [sharpening, setSharpening] = useState('0');
  const [saturation, setSaturation] = useState('0');
  const [ppi, setPpi] = useState('300');

  const [settingsTab, setSettingsTab] = useState<SettingsTab>('engine');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const batchInputRef = useRef<HTMLInputElement>(null);
  const sliderContainerRef = useRef<HTMLDivElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResultUrl(null);
    setErrorMsg(null);
  };

  const handleBatchSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files) return;
    setBatchFiles(Array.from(event.target.files));
    setResultUrl(null);
    setErrorMsg(null);
  };

  // Slider Drag Interactions (Pointer & Touch friendly)
  const updateSliderPosition = (clientX: number) => {
    if (!sliderContainerRef.current) return;
    const rect = sliderContainerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(Math.round(percentage));
  };

  const handleSliderPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    setIsDraggingSlider(true);
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {}
    updateSliderPosition(e.clientX);
  };

  const handleSliderPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDraggingSlider) return;
    updateSliderPosition(e.clientX);
  };

  const handleSliderPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    setIsDraggingSlider(false);
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {}
  };

  const handleSliderKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'ArrowLeft') {
      setSliderPos((prev) => Math.max(0, prev - 5));
    } else if (e.key === 'ArrowRight') {
      setSliderPos((prev) => Math.min(100, prev + 5));
    }
  };

  const handleProcess = async () => {
    if (tab === 'single' && !file) return;
    if (tab === 'batch' && batchFiles.length === 0) return;
    setIsProcessing(true);
    const formData = new FormData();
    formData.append('ai_model', model);
    formData.append('face_restoration', faceRestoration.toString());
    formData.append('skin_smoothing', skinSmoothing);
    formData.append('preserve_colors', preserveColors.toString());
    formData.append('tile_size', tileSize);
    formData.append('sharpening_strength', sharpening);
    formData.append('saturation_adjustment', saturation);
    formData.append('output_ppi', ppi);
    formData.append('jpeg_quality', '95');
    formData.append('restoration_task', restorationTask);
    formData.append('old_photo_mode', oldPhotoMode.toString());
    formData.append('passes', passes);
    formData.append('capture_mode', captureMode);

    try {
      let response: Response;
      if (tab === 'single') {
        formData.append('image', file as Blob);
        response = await fetch(`${API_URL}/api/enhance/single`, {
          method: 'POST',
          body: formData,
        });
      } else {
        batchFiles.forEach((selectedFile) =>
          formData.append('images', selectedFile)
        );
        response = await fetch(`${API_URL}/api/enhance/batch`, {
          method: 'POST',
          body: formData,
        });
      }
      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(
          errorData?.detail || 'Processing failed. Please check server logs.'
        );
      }
      setResultUrl(URL.createObjectURL(await response.blob()));
      setErrorMsg(null);
    } catch (error) {
      setErrorMsg(
        error instanceof Error ? error.message : 'An unexpected error occurred.'
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (!resultUrl) return;
    const link = document.createElement('a');
    link.href = resultUrl;
    link.download =
      tab === 'single' ? `enhanced_${file?.name || 'photo.jpg'}` : 'batch_enhanced.zip';
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Hybrid Photo Enhancer</h1>
        <p>Classical Computer Vision &amp; Deep Learning Restoration</p>
      </header>

      {/* Main Mode Tabs */}
      <div className="tabs">
        <button
          type="button"
          className={`tab-btn ${tab === 'single' ? 'active' : ''}`}
          onClick={() => setTab('single')}
        >
          <ImageIcon size={18} /> Single Photo
        </button>
        <button
          type="button"
          className={`tab-btn ${tab === 'batch' ? 'active' : ''}`}
          onClick={() => setTab('batch')}
        >
          <Layers size={18} /> Batch Processing
        </button>
      </div>

      <div className="main-content">
        {/* Workspace Card */}
        <div className="workspace glass-panel">
          {errorMsg && (
            <div className="error-banner">
              <AlertTriangle size={18} />
              <span>{errorMsg}</span>
              <button
                type="button"
                className="error-close"
                onClick={() => setErrorMsg(null)}
                aria-label="Dismiss error"
              >
                <X size={16} />
              </button>
            </div>
          )}

          {tab === 'single' ? (
            <div>
              {!previewUrl ? (
                <div
                  className="upload-area"
                  role="button"
                  tabIndex={0}
                  onClick={() => fileInputRef.current?.click()}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ')
                      fileInputRef.current?.click();
                  }}
                >
                  <Upload size={44} className="upload-icon" />
                  <h3>Upload a photo</h3>
                  <p className="info-text">
                    Drag &amp; drop or click to browse image
                  </p>
                  <input
                    type="file"
                    className="file-input"
                    ref={fileInputRef}
                    accept="image/*"
                    onChange={handleFileSelect}
                  />
                </div>
              ) : (
                <div className="image-preview-container">
                  {!resultUrl ? (
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="preview-image"
                    />
                  ) : (
                    <div
                      className="slider-container"
                      ref={sliderContainerRef}
                      onPointerDown={handleSliderPointerDown}
                      onPointerMove={handleSliderPointerMove}
                      onPointerUp={handleSliderPointerUp}
                      onPointerCancel={handleSliderPointerUp}
                      onKeyDown={handleSliderKeyDown}
                      tabIndex={0}
                      role="slider"
                      aria-label="Before and after image comparison slider"
                      aria-valuenow={sliderPos}
                      aria-valuemin={0}
                      aria-valuemax={100}
                    >
                      <span className="slider-badge slider-badge-left">
                        Original
                      </span>
                      <span className="slider-badge slider-badge-right">
                        Enhanced
                      </span>

                      {/* Original Base Image */}
                      <img
                        src={previewUrl}
                        alt="Original"
                        className="slider-img"
                        draggable={false}
                      />

                      {/* Enhanced Overlaid Image with dynamic clip-path */}
                      <img
                        src={resultUrl}
                        alt="Enhanced"
                        className="slider-img-overlay"
                        style={{ clipPath: `inset(0 0 0 ${sliderPos}%)` }}
                        draggable={false}
                      />

                      {/* Draggable vertical divider handle */}
                      <div
                        className="slider-handle"
                        style={{ left: `${sliderPos}%` }}
                      />
                    </div>
                  )}

                  <div className="preview-actions">
                    <button
                      type="button"
                      className="btn-secondary"
                      onClick={() => {
                        setFile(null);
                        setPreviewUrl(null);
                        setResultUrl(null);
                      }}
                    >
                      Change Photo
                    </button>
                    {resultUrl && (
                      <button
                        type="button"
                        className="btn-primary"
                        onClick={handleDownload}
                      >
                        <Download size={18} /> Download Result
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div>
              <div
                className="upload-area"
                role="button"
                tabIndex={0}
                onClick={() => batchInputRef.current?.click()}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ')
                    batchInputRef.current?.click();
                }}
              >
                <FolderUp size={44} className="upload-icon" />
                <h3>Upload Multiple Photos</h3>
                <p className="info-text">
                  {batchFiles.length > 0
                    ? `${batchFiles.length} files selected`
                    : 'Select folder or multiple images to batch process'}
                </p>
                <input
                  type="file"
                  className="file-input"
                  ref={batchInputRef}
                  accept="image/*"
                  multiple
                  onChange={handleBatchSelect}
                />
              </div>

              {resultUrl && (
                <div className="batch-download">
                  <button
                    type="button"
                    className="btn-primary"
                    onClick={handleDownload}
                  >
                    <Download size={18} /> Download Batch ZIP
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Processing Settings Panel */}
        <div className="settings-panel glass-panel">
          <div className="settings-heading">
            <div className="settings-title-wrap">
              <Zap size={18} color="var(--accent-solid)" />
              <h3>Settings</h3>
            </div>
            <span className="settings-status-pill">
              {tab === 'single' ? 'Single' : 'Batch'}
            </span>
          </div>

          <div
            className="settings-tabs"
            role="tablist"
            aria-label="Processing settings navigation"
          >
            <button
              type="button"
              className={`settings-tab ${
                settingsTab === 'engine' ? 'active' : ''
              }`}
              onClick={() => setSettingsTab('engine')}
              role="tab"
              aria-selected={settingsTab === 'engine'}
            >
              <Zap size={14} /> Engine
            </button>
            <button
              type="button"
              className={`settings-tab ${
                settingsTab === 'restoration' ? 'active' : ''
              }`}
              onClick={() => setSettingsTab('restoration')}
              role="tab"
              aria-selected={settingsTab === 'restoration'}
            >
              <ImageIcon size={14} /> Restore
            </button>
            <button
              type="button"
              className={`settings-tab ${
                settingsTab === 'finishing' ? 'active' : ''
              }`}
              onClick={() => setSettingsTab('finishing')}
              role="tab"
              aria-selected={settingsTab === 'finishing'}
            >
              <Sliders size={14} /> Finish
            </button>
          </div>

          <div className="settings-content">
            {settingsTab === 'engine' && (
              <>
                <div className="setting-group">
                  <div className="setting-label-row">
                    <label htmlFor="model-select">AI Super-Resolution</label>
                  </div>
                  <select
                    id="model-select"
                    value={model}
                    onChange={(event) => setModel(event.target.value)}
                  >
                    <option value="RealESRGAN x2">RealESRGAN x2 (2× Upscale)</option>
                    <option value="RealESRGAN x4">RealESRGAN x4 (4× Deep Detail)</option>
                    <option value="RealESRGAN Anime x4">RealESRGAN Anime x4 (Art / Illustration)</option>
                    <option value="None">None (Skip Upscaling - Keep Original Resolution)</option>
                  </select>
                </div>

                <div className="setting-group">
                  <div className="setting-label-row">
                    <label htmlFor="capture-select">Lens &amp; Distortion</label>
                    <span className="info-text">Capture type</span>
                  </div>
                  <select
                    id="capture-select"
                    value={captureMode}
                    onChange={(event) => setCaptureMode(event.target.value)}
                  >
                    <option value="Standard">None (Skip Distortion Correction)</option>
                    <option value="Wide-angle Distortion Correction">Wide-angle Distortion Correction (24mm Barrel Fix)</option>
                    <option value="Digital 2x Quality Recovery">Digital 2x Quality Recovery (Noise &amp; Softness Fix)</option>
                  </select>
                </div>

                <div className="setting-group">
                  <div className="setting-label-row">
                    <label htmlFor="passes-select">Refinement Passes</label>
                  </div>
                  <select
                    id="passes-select"
                    value={passes}
                    onChange={(event) => setPasses(event.target.value)}
                    disabled={model === 'None'}
                  >
                    <option value="1">1 Pass (Standard)</option>
                    <option value="2">2 Passes (Deep Detail)</option>
                    <option value="3">3 Passes (Ultra Detail)</option>
                  </select>
                </div>

                <div className="setting-group">
                  <div className="setting-label-row">
                    <label htmlFor="tile-select">Tile Size</label>
                    <span className="info-text">VRAM usage</span>
                  </div>
                  <select
                    id="tile-select"
                    value={tileSize}
                    onChange={(event) => setTileSize(event.target.value)}
                    disabled={model === 'None'}
                  >
                    <option value="128">128 (Low VRAM)</option>
                    <option value="256">256 (Default)</option>
                    <option value="512">512 (Fast / High VRAM)</option>
                  </select>
                </div>
              </>
            )}

            {settingsTab === 'restoration' && (
              <>
                <div className="setting-group">
                  <div className="setting-label-row">
                    <label htmlFor="restoration-select">
                      Restormer Deblur
                    </label>
                  </div>
                  <select
                    id="restoration-select"
                    value={restorationTask}
                    onChange={(event) =>
                      setRestorationTask(event.target.value)
                    }
                  >
                    <option value="None">None (Skip Deblurring)</option>
                    <option value="Motion_Deblurring">
                      Motion Deblurring
                    </option>
                    <option value="Single_Image_Defocus_Deblurring">
                      Defocus Deblurring
                    </option>
                  </select>
                </div>

                <div className="toggle-row">
                  <div className="toggle-info">
                    <span className="toggle-title">
                      GFPGAN Face Restoration
                    </span>
                    <span className="toggle-subtitle">
                      Reconstruct facial landmarks
                    </span>
                  </div>
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={faceRestoration}
                      onChange={(event) =>
                        setFaceRestoration(event.target.checked)
                      }
                      aria-label="Toggle GFPGAN Face Restoration"
                    />
                    <span className="toggle-slider" />
                  </label>
                </div>

                <div className="setting-group">
                  <div className="setting-label-row">
                    <label>Skin Smoothing</label>
                    <span
                      className={`setting-badge ${
                        faceRestoration ? 'accent' : ''
                      }`}
                    >
                      {faceRestoration ? skinSmoothing : 'Off'}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={skinSmoothing}
                    onChange={(event) => setSkinSmoothing(event.target.value)}
                    disabled={!faceRestoration}
                    aria-label="Skin smoothing intensity"
                  />
                </div>

                <div className="toggle-row">
                  <div className="toggle-info">
                    <span className="toggle-title">
                      Preserve Original Colors
                    </span>
                    <span className="toggle-subtitle">
                      Match vintage photo palette
                    </span>
                  </div>
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={preserveColors}
                      onChange={(event) =>
                        setPreserveColors(event.target.checked)
                      }
                      disabled={!faceRestoration}
                      aria-label="Toggle Preserve Original Colors"
                    />
                    <span className="toggle-slider" />
                  </label>
                </div>

                <div className="toggle-row">
                  <div className="toggle-info">
                    <span className="toggle-title">Old Photo Mode</span>
                    <span className="toggle-subtitle">
                      Classical denoise + sharpen
                    </span>
                  </div>
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={oldPhotoMode}
                      onChange={(event) =>
                        setOldPhotoMode(event.target.checked)
                      }
                      aria-label="Toggle Old Photo Mode"
                    />
                    <span className="toggle-slider" />
                  </label>
                </div>
              </>
            )}

            {settingsTab === 'finishing' && (
              <>
                <div className="setting-group">
                  <div className="setting-label-row">
                    <label>Sharpening Strength</label>
                    <span className="setting-badge">{sharpening}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="0.5"
                    step="0.05"
                    value={sharpening}
                    onChange={(event) => setSharpening(event.target.value)}
                    aria-label="Sharpening strength"
                  />
                </div>

                <div className="setting-group">
                  <div className="setting-label-row">
                    <label>Saturation Adjustment</label>
                    <span className="setting-badge">
                      {Number(saturation) > 0 ? `+${saturation}` : saturation}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="-0.3"
                    max="0.3"
                    step="0.05"
                    value={saturation}
                    onChange={(event) => setSaturation(event.target.value)}
                    aria-label="Saturation adjustment"
                  />
                </div>

                <div className="setting-group">
                  <div className="setting-label-row">
                    <label htmlFor="ppi-select">Output Print PPI</label>
                  </div>
                  <select
                    id="ppi-select"
                    value={ppi}
                    onChange={(event) => setPpi(event.target.value)}
                  >
                    <option value="72">72 PPI (Web &amp; Screen)</option>
                    <option value="300">300 PPI (High Quality Print)</option>
                    <option value="600">600 PPI (Ultra-Fine Print)</option>
                  </select>
                </div>
              </>
            )}
          </div>

          <div className="action-area">
            <button
              type="button"
              className="btn-primary"
              onClick={handleProcess}
              disabled={
                isProcessing ||
                (tab === 'single' && !file) ||
                (tab === 'batch' && batchFiles.length === 0)
              }
            >
              {isProcessing ? (
                <>
                  <Sparkles size={18} className="loader" /> Processing...
                </>
              ) : (
                <>
                  <Sparkles size={18} /> Enhance Now
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
