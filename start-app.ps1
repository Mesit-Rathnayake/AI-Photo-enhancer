$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$frontend = Join-Path $root 'frontend'

if (-not (Test-Path $python)) {
    throw "Python environment not found at $python. Create the .venv first."
}

function Test-HttpReady([string]$url) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 2
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    }
    catch {
        return $false
    }
}

if (-not (Test-HttpReady 'http://127.0.0.1:8000/docs')) {
    Start-Process powershell.exe -WorkingDirectory $root -ArgumentList @(
        '-NoExit',
        '-ExecutionPolicy', 'Bypass',
        '-Command', "& '$python' -m uvicorn api:app --host 0.0.0.0 --port 8000"
    ) | Out-Null
}

if (-not (Test-HttpReady 'http://127.0.0.1:5173/')) {
    Start-Process powershell.exe -WorkingDirectory $frontend -ArgumentList @(
        '-NoExit',
        '-ExecutionPolicy', 'Bypass',
        '-Command', 'npm.cmd run dev -- --host 0.0.0.0 --port 5173'
    ) | Out-Null
}

$deadline = (Get-Date).AddSeconds(30)
do {
    $backendReady = Test-HttpReady 'http://127.0.0.1:8000/docs'
    $frontendReady = Test-HttpReady 'http://127.0.0.1:5173/'

    if ($backendReady -and $frontendReady) {
        Start-Process 'http://localhost:5173/'
        Write-Host 'AI Photo Enhancer is running at http://localhost:5173/' -ForegroundColor Green
        Write-Host 'API documentation is available at http://localhost:8000/docs'
        exit 0
    }

    Start-Sleep -Milliseconds 500
} while ((Get-Date) -lt $deadline)

throw "The application did not become ready within 30 seconds. Check the backend and frontend windows for errors."