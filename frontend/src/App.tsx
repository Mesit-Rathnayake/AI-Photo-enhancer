import { useRef, useState } from 'react';
import { AlertTriangle, Download, FolderUp, ImageIcon, Layers, Sliders, Sparkles, Upload, X, Zap } from 'lucide-react';
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
  const [model, setModel] = useState('RealESRGAN x2');
  const [faceRestoration, setFaceRestoration] = useState(false);
  const [skinSmoothing, setSkinSmoothing] = useState('0.2');
  const [preserveColors, setPreserveColors] = useState(true);
  const [tileSize, setTileSize] = useState('256');
  const [sharpening, setSharpening] = useState('0');
  const [saturation, setSaturation] = useState('0');
  const [ppi, setPpi] = useState('300');
  const [restorationTask, setRestorationTask] = useState('None');
  const [oldPhotoMode, setOldPhotoMode] = useState(false);
  const [passes, setPasses] = useState('1');
  const [captureMode, setCaptureMode] = useState('Standard');
  const [settingsTab, setSettingsTab] = useState<SettingsTab>('engine');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const batchInputRef = useRef<HTMLInputElement>(null);

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
        response = await fetch(`${API_URL}/api/enhance/single`, { method: 'POST', body: formData });
      } else {
        batchFiles.forEach(selectedFile => formData.append('images', selectedFile));
        response = await fetch(`${API_URL}/api/enhance/batch`, { method: 'POST', body: formData });
      }
      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || 'Processing failed. Please try different settings.');
      }
      setResultUrl(URL.createObjectURL(await response.blob()));
      setErrorMsg(null);
    } catch (error) {
      setErrorMsg(error instanceof Error ? error.message : 'An unexpected error occurred.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (!resultUrl) return;
    const link = document.createElement('a');
    link.href = resultUrl;
    link.download = tab === 'single' ? `enhanced_${file?.name}` : 'batch_enhanced.zip';
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>AI Photo Enhancer Pro</h1>
        <p>Premium AI Super-Resolution &amp; Face Restoration</p>
      </header>

      <div className="tabs">
        <button type="button" className={`tab-btn ${tab === 'single' ? 'active' : ''}`} onClick={() => setTab('single')}><ImageIcon size={20} /> Single Photo</button>
        <button type="button" className={`tab-btn ${tab === 'batch' ? 'active' : ''}`} onClick={() => setTab('batch')}><Layers size={20} /> Batch Processing</button>
      </div>

      <div className="main-content">
        <div className="workspace glass-panel">
          {errorMsg && <div className="error-banner"><AlertTriangle size={20} /><span>{errorMsg}</span><button type="button" className="error-close" onClick={() => setErrorMsg(null)}><X size={16} /></button></div>}
          {tab === 'single' ? (
            <div style={{ padding: '2rem' }}>
              {!previewUrl ? (
                <div className="upload-area" role="button" tabIndex={0} onClick={() => fileInputRef.current?.click()} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') fileInputRef.current?.click(); }}>
                  <Upload size={48} className="upload-icon" /><h3>Upload a photo</h3><p className="info-text">Drag and drop or click to browse</p>
                  <input type="file" className="file-input" ref={fileInputRef} accept="image/*" onChange={handleFileSelect} />
                </div>
              ) : (
                <div className="image-preview-container">
                  {!resultUrl ? <img src={previewUrl} alt="Preview" className="preview-image" /> : <div className="slider-container"><img src={previewUrl} alt="Original" className="slider-img" /><img src={resultUrl} alt="Enhanced" className="slider-img-overlay" style={{ clipPath: `inset(0 0 0 ${sliderPos}%)` }} /><div className="slider-handle" style={{ left: `${sliderPos}%` }} /><input type="range" min="0" max="100" value={sliderPos} onChange={event => setSliderPos(Number(event.target.value))} className="slider-input" aria-label="Before and after comparison" /></div>}
                  <div className="preview-actions"><button type="button" className="btn-secondary" onClick={() => { setFile(null); setPreviewUrl(null); setResultUrl(null); }}>Change Photo</button>{resultUrl && <button type="button" className="btn-primary" onClick={handleDownload}><Download size={18} /> Download</button>}</div>
                </div>
              )}
            </div>
          ) : (
            <div style={{ padding: '2rem' }}><div className="upload-area" role="button" tabIndex={0} onClick={() => batchInputRef.current?.click()} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') batchInputRef.current?.click(); }}><FolderUp size={48} className="upload-icon" /><h3>Upload Multiple Photos</h3><p className="info-text">{batchFiles.length > 0 ? `${batchFiles.length} files selected` : 'Select folder or multiple files'}</p><input type="file" className="file-input" ref={batchInputRef} accept="image/*" multiple onChange={handleBatchSelect} /></div>{resultUrl && <div className="batch-download"><button type="button" className="btn-primary" onClick={handleDownload}><Download size={18} /> Download ZIP</button></div>}</div>
          )}
        </div>

        <div className="settings-panel glass-panel">
          <div className="settings-heading"><h3><Zap size={20} color="var(--accent-solid)" /> Processing Settings</h3><span className="settings-status">{tab === 'single' ? 'Single photo' : 'Batch queue'}</span></div>
          <div className="settings-tabs" role="tablist" aria-label="Processing settings">
            <button type="button" className={`settings-tab ${settingsTab === 'engine' ? 'active' : ''}`} onClick={() => setSettingsTab('engine')} role="tab" aria-selected={settingsTab === 'engine'}><Zap size={16} /> Engine</button>
            <button type="button" className={`settings-tab ${settingsTab === 'restoration' ? 'active' : ''}`} onClick={() => setSettingsTab('restoration')} role="tab" aria-selected={settingsTab === 'restoration'}><ImageIcon size={16} /> Restore</button>
            <button type="button" className={`settings-tab ${settingsTab === 'finishing' ? 'active' : ''}`} onClick={() => setSettingsTab('finishing')} role="tab" aria-selected={settingsTab === 'finishing'}><Sliders size={16} /> Finish</button>
          </div>

          <div className="settings-content">
            {settingsTab === 'engine' && <><div className="setting-group"><label>Model</label><select value={model} onChange={event => setModel(event.target.value)}><option>RealESRGAN x2</option><option>RealESRGAN x4</option><option>RealESRGAN Anime x4</option></select></div><div className="setting-group"><label>Camera Correction <span className="info-text">Capture type</span></label><select value={captureMode} onChange={event => setCaptureMode(event.target.value)}><option>Standard</option><option>Digital 2x Quality Recovery</option><option>1x Lens Correction</option><option>Wide-angle Portrait Correction</option><option>Radial Lens Distortion Correction</option></select></div><div className="setting-group"><label>Refinement Passes</label><select value={passes} onChange={event => setPasses(event.target.value)}><option value="1">1 Pass (Standard)</option><option value="2">2 Passes (Deep Detail)</option><option value="3">3 Passes (Ultra Detail)</option></select></div><div className="setting-group"><label>Tile Size <span className="info-text">VRAM usage</span></label><select value={tileSize} onChange={event => setTileSize(event.target.value)}><option value="128">128 (Low VRAM)</option><option value="256">256 (Default)</option><option value="512">512 (Fast/High VRAM)</option></select></div></>}
            {settingsTab === 'restoration' && <><div className="setting-group"><label>Photo Restoration</label><select value={restorationTask} onChange={event => setRestorationTask(event.target.value)}><option value="None">None (Skip)</option><option value="Motion_Deblurring">Motion Deblurring</option><option value="Single_Image_Defocus_Deblurring">Defocus Deblurring</option></select></div><div className="setting-group checkbox"><label>GFPGAN Face Restoration</label><input type="checkbox" checked={faceRestoration} onChange={event => setFaceRestoration(event.target.checked)} /></div><div className="setting-group"><label>Skin Smoothing: {skinSmoothing}</label><input type="range" min="0" max="1" step="0.05" value={skinSmoothing} onChange={event => setSkinSmoothing(event.target.value)} disabled={!faceRestoration} /></div><div className="setting-group checkbox"><label>Preserve Original Colors</label><input type="checkbox" checked={preserveColors} onChange={event => setPreserveColors(event.target.checked)} disabled={!faceRestoration} /></div><div className="setting-group checkbox"><label>Old Photo Mode <span className="info-text">Denoise + sharpen</span></label><input type="checkbox" checked={oldPhotoMode} onChange={event => setOldPhotoMode(event.target.checked)} /></div></>}
            {settingsTab === 'finishing' && <><div className="setting-group"><label>Sharpening: {sharpening}</label><input type="range" min="0" max="0.5" step="0.05" value={sharpening} onChange={event => setSharpening(event.target.value)} /></div><div className="setting-group"><label>Saturation: {saturation}</label><input type="range" min="-0.3" max="0.3" step="0.05" value={saturation} onChange={event => setSaturation(event.target.value)} /></div><div className="setting-group"><label>Output PPI</label><select value={ppi} onChange={event => setPpi(event.target.value)}><option value="72">72 (Web)</option><option value="300">300 (Print)</option><option value="600">600 (High-Res Print)</option></select></div></>}
          </div>

          <div className="action-area"><button type="button" className="btn-primary" onClick={handleProcess} disabled={isProcessing || (tab === 'single' && !file) || (tab === 'batch' && batchFiles.length === 0)}>{isProcessing ? <><Sparkles size={20} className="loader" /> Processing...</> : <><Sparkles size={20} /> Enhance Now</>}</button></div>
        </div>
      </div>
    </div>
  );
}

export default App;
