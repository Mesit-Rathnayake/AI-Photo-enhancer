import { useState, useRef } from 'react';
import { Upload, ImageIcon, Layers, Zap, Sliders, Download, Sparkles, FolderUp, AlertTriangle, X } from 'lucide-react';
import './App.css';

const API_URL = 'http://127.0.0.1:8000';

function App() {
  const [tab, setTab] = useState<'single' | 'batch'>('single');
  const [file, setFile] = useState<File | null>(null);
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [sliderPos, setSliderPos] = useState(50);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  
  // Settings State
  const [model, setModel] = useState('RealESRGAN x2');
  const [faceRestoration, setFaceRestoration] = useState(false);
  const [skinSmoothing, setSkinSmoothing] = useState('0.2');
  const [preserveColors, setPreserveColors] = useState(true);
  const [tileSize, setTileSize] = useState('256');
  const [sharpening, setSharpening] = useState('0');
  const [saturation, setSaturation] = useState('0');
  const [ppi, setPpi] = useState('300');
  const [jpegQuality] = useState('95');
  const [restorationTask, setRestorationTask] = useState('None');
  const [oldPhotoMode, setOldPhotoMode] = useState(false);
  const [passes, setPasses] = useState('1');
  const [captureMode, setCaptureMode] = useState('Standard');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const batchInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setResultUrl(null);
    }
  };

  const handleBatchSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setBatchFiles(Array.from(e.target.files));
      setResultUrl(null);
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
    formData.append('jpeg_quality', jpegQuality);
    formData.append('restoration_task', restorationTask);
    formData.append('old_photo_mode', oldPhotoMode.toString());
    formData.append('passes', passes);
    formData.append('capture_mode', captureMode);

    try {
      let response;
      if (tab === 'single') {
        formData.append('image', file as Blob);
        response = await fetch(`${API_URL}/api/enhance/single`, {
          method: 'POST',
          body: formData,
        });
      } else {
        batchFiles.forEach(f => formData.append('images', f));
        response = await fetch(`${API_URL}/api/enhance/batch`, {
          method: 'POST',
          body: formData,
        });
      }

      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        const detail = errData?.detail || 'Processing failed. Please try different settings.';
        throw new Error(detail);
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setResultUrl(url);
      setErrorMsg(null);

    } catch (error: any) {
      console.error(error);
      setErrorMsg(error.message || "An unexpected error occurred.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (!resultUrl) return;
    const a = document.createElement('a');
    a.href = resultUrl;
    a.download = tab === 'single' ? `enhanced_${file?.name}` : 'batch_enhanced.zip';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>✨ AI Photo Enhancer Pro</h1>
        <p>Premium AI Super-Resolution & Face Restoration</p>
      </header>

      <div className="tabs">
        <button 
          className={`tab-btn ${tab === 'single' ? 'active' : ''}`}
          onClick={() => setTab('single')}
        >
          <ImageIcon size={20} /> Single Photo
        </button>
        <button 
          className={`tab-btn ${tab === 'batch' ? 'active' : ''}`}
          onClick={() => setTab('batch')}
        >
          <Layers size={20} /> Batch Processing
        </button>
      </div>

      <div className="main-content">
        <div className="workspace glass-panel">
          {errorMsg && (
            <div className="error-banner">
              <AlertTriangle size={20} />
              <span>{errorMsg}</span>
              <button className="error-close" onClick={() => setErrorMsg(null)}><X size={16} /></button>
            </div>
          )}
          {tab === 'single' ? (
            <div style={{ padding: '2rem' }}>
              {!previewUrl ? (
                <div className="upload-area" onClick={() => fileInputRef.current?.click()}>
                  <Upload size={48} className="upload-icon" />
                  <h3>Upload a photo</h3>
                  <p className="info-text">Drag and drop or click to browse</p>
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
                    <img src={previewUrl} alt="Preview" className="preview-image" />
                  ) : (
                    <div className="slider-container">
                      <img src={previewUrl} alt="Original" className="slider-img" />
                      <img 
                        src={resultUrl} 
                        alt="Enhanced" 
                        className="slider-img-overlay"
                        style={{ clipPath: `inset(0 0 0 ${sliderPos}%)` }}
                      />
                      <div className="slider-handle" style={{ left: `${sliderPos}%` }} />
                      <input 
                        type="range" 
                        min="0" max="100" 
                        value={sliderPos} 
                        onChange={(e) => setSliderPos(Number(e.target.value))}
                        className="slider-input"
                      />
                    </div>
                  )}
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <button className="btn-secondary" onClick={() => { setFile(null); setPreviewUrl(null); setResultUrl(null); }}>
                      Change Photo
                    </button>
                    {resultUrl && (
                      <button className="btn-primary" onClick={handleDownload}>
                        <Download size={18} /> Download
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div style={{ padding: '2rem' }}>
              <div className="upload-area" onClick={() => batchInputRef.current?.click()}>
                <FolderUp size={48} className="upload-icon" />
                <h3>Upload Multiple Photos</h3>
                <p className="info-text">{batchFiles.length > 0 ? `${batchFiles.length} files selected` : 'Select folder or multiple files'}</p>
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
                <div style={{ marginTop: '2rem', textAlign: 'center' }}>
                  <button className="btn-primary" style={{ margin: '0 auto' }} onClick={handleDownload}>
                    <Download size={18} /> Download ZIP
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="settings-panel glass-panel">
          <h3><Zap size={20} color="var(--accent-solid)" /> AI Engine</h3>
          
          <div className="setting-group">
            <label>Model</label>
            <select value={model} onChange={e => setModel(e.target.value)}>
              <option value="RealESRGAN x2">RealESRGAN x2 (Balanced)</option>
              <option value="RealESRGAN x4">RealESRGAN x4 (Max Detail)</option>
              <option value="RealESRGAN Anime x4">Anime x4 (Illustrations)</option>
            </select>
          </div>

          <div className="setting-group">
            <label>Camera Correction <span className="info-text">Choose by capture type</span></label>
            <select value={captureMode} onChange={e => setCaptureMode(e.target.value)}>
              <option value="Standard">Standard processing</option>
              <option value="Digital 2x Quality Recovery">Digital 2x quality recovery</option>
              <option value="1x Lens Correction">1x lens correction</option>
              <option value="Wide-angle Portrait Correction">Wide-angle portrait correction</option>
              <option value="Radial Lens Distortion Correction">Radial lens distortion correction</option>
            </select>
          </div>

          <div className="setting-group">
            <label>Photo Restoration <span className="info-text">Old photos</span></label>
            <select value={restorationTask} onChange={e => setRestorationTask(e.target.value)}>
              <option value="None">None (Skip)</option>
              <option value="Motion_Deblurring">Motion Deblurring</option>
              <option value="Single_Image_Defocus_Deblurring">Defocus Deblurring</option>
            </select>
          </div>

          <div className="setting-group">
            <label>Refinement Passes <span className="info-text">Reconstruct Details</span></label>
            <select value={passes} onChange={e => setPasses(e.target.value)}>
              <option value="1">1 Pass (Standard)</option>
              <option value="2">2 Passes (Deep Detail)</option>
              <option value="3">3 Passes (Ultra Detail)</option>
            </select>
          </div>

          <div className="setting-group checkbox">
            <label>Preserve Original Colors</label>
            <input
              type="checkbox"
              checked={preserveColors}
              onChange={e => setPreserveColors(e.target.checked)}
              disabled={!faceRestoration}
            />
          </div>

          <div className="setting-group checkbox">
            <label>GFPGAN Face Restoration</label>
            <input 
              type="checkbox" 
              checked={faceRestoration}
              onChange={e => setFaceRestoration(e.target.checked)}
            />
          </div>

          <div className="setting-group">
            <label>Skin Smoothing: {skinSmoothing}</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={skinSmoothing}
              onChange={e => setSkinSmoothing(e.target.value)}
              disabled={!faceRestoration}
            />
          </div>

          <div className="setting-group checkbox">
            <label>🕰️ Old Photo Mode <span className="info-text">Denoise + Sharpen</span></label>
            <input 
              type="checkbox" 
              checked={oldPhotoMode}
              onChange={e => setOldPhotoMode(e.target.checked)}
            />
          </div>

          <div className="setting-group">
            <label>Tile Size <span className="info-text">VRAM usage</span></label>
            <select value={tileSize} onChange={e => setTileSize(e.target.value)}>
              <option value="128">128 (Low VRAM)</option>
              <option value="256">256 (Default)</option>
              <option value="512">512 (Fast/High VRAM)</option>
            </select>
          </div>

          <h3 style={{ marginTop: '1rem' }}><Sliders size={20} color="var(--accent-solid)" /> Finishing</h3>

          <div className="setting-group">
            <label>Sharpening: {sharpening}</label>
            <input 
              type="range" min="0" max="0.5" step="0.05" 
              value={sharpening} onChange={e => setSharpening(e.target.value)}
            />
          </div>

          <div className="setting-group">
            <label>Saturation: {saturation}</label>
            <input 
              type="range" min="-0.3" max="0.3" step="0.05" 
              value={saturation} onChange={e => setSaturation(e.target.value)}
            />
          </div>
          
          <div className="setting-group">
            <label>Output PPI</label>
            <select value={ppi} onChange={e => setPpi(e.target.value)}>
              <option value="72">72 (Web)</option>
              <option value="300">300 (Print)</option>
              <option value="600">600 (High-Res Print)</option>
            </select>
          </div>

          <div className="action-area">
            <button 
              className="btn-primary" 
              onClick={handleProcess}
              disabled={isProcessing || (tab === 'single' && !file) || (tab === 'batch' && batchFiles.length === 0)}
            >
              {isProcessing ? (
                <><Sparkles size={20} className="loader" /> Processing...</>
              ) : (
                <><Sparkles size={20} /> Enhance Now</>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
