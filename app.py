"""Ultra-Modern & Responsive Web Application for AI YouTube Shorts Generator.

Built with FastAPI, Tailwind CSS, Lucide Icons, and Server-Sent Events (SSE).
Usage:
    python app.py
"""
import os
import re
import sys
import json
import asyncio
import threading
import traceback
from pathlib import Path
from typing import Optional, List, Dict

try:
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print(
        "FastAPI and Uvicorn are required. Install with:\n"
        "    pip install fastapi uvicorn python-multipart"
    )
    sys.exit(1)

from shorts_generator import generate_shorts
from shorts_generator.config import LOCAL_OUTPUT_DIR

app = FastAPI(title="AI YouTube Shorts Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory progress tracker for real-time SSE updates
job_status: Dict[str, Dict] = {}


def _extract_youtube_id(url: str) -> Optional[str]:
    if not url:
        return None
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"embed\/([0-9A-Za-z_-]{11})",
        r"shorts\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url.strip())
        if match:
            return match.group(1)
    return None


@app.get("/api/video/{filename}")
async def get_video(filename: str):
    """Serve rendered short mp4 files."""
    file_path = os.path.join(LOCAL_OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)


@app.post("/api/generate")
async def generate_api(
    url: Optional[str] = Form(None),
    mode: str = Form("local"),
    video_mode: str = Form("crop-face"),
    aspect_ratio: str = Form("9:16"),
    num_clips: int = Form(3),
    use_manual_time: bool = Form(False),
    start_time: Optional[float] = Form(None),
    end_time: Optional[float] = Form(None),
    language: Optional[str] = Form(None),
    download_format: str = Form("720"),
    file: Optional[UploadFile] = File(None),
):
    """Start shorts generation."""
    source_path = ""
    if file and file.filename:
        upload_dir = os.path.join(LOCAL_OUTPUT_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        dest_path = os.path.join(upload_dir, file.filename)
        content = await file.read()
        with open(dest_path, "wb") as f:
            f.write(content)
        source_path = dest_path
    elif url and url.strip():
        source_path = url.strip()
    else:
        raise HTTPException(status_code=400, detail="Please provide a YouTube URL or upload a video file.")

    st = float(start_time) if (use_manual_time and start_time is not None and start_time >= 0) else None
    et = float(end_time) if (use_manual_time and end_time is not None and end_time > 0) else None

    if use_manual_time and st is not None and et is not None and et <= st:
        raise HTTPException(status_code=400, detail="End time must be greater than Start time.")

    lang = language.strip() if (language and language.strip() and language.strip().lower() != "auto") else None

    try:
        result = generate_shorts(
            youtube_url=source_path,
            num_clips=int(num_clips),
            aspect_ratio=aspect_ratio,
            download_format=download_format,
            language=lang,
            mode=mode.lower(),
            start_time=st,
            end_time=et,
            video_mode=video_mode.lower(),
        )

        # Normalize clip paths to web-accessible URLs
        shorts_data = []
        for i, s in enumerate(result.get("shorts", []), 1):
            clip_path = s.get("clip_url")
            web_url = None
            if clip_path and os.path.exists(clip_path):
                filename = os.path.basename(clip_path)
                web_url = f"/api/video/{filename}"
            elif clip_path and clip_path.startswith("http"):
                web_url = clip_path

            shorts_data.append({
                "id": i,
                "title": s.get("title", f"Short #{i}"),
                "hook": s.get("hook_sentence", ""),
                "score": s.get("score", 100),
                "start_time": s.get("start_time", 0.0),
                "end_time": s.get("end_time", 0.0),
                "duration": round(s.get("end_time", 0.0) - s.get("start_time", 0.0), 1),
                "clip_url": web_url,
                "error": s.get("error"),
            })

        return {
            "success": True,
            "mode": result.get("mode", mode),
            "source_video": result.get("source_video_url"),
            "shorts": shorts_data,
            "raw": result,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return HTML_CONTENT


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI YouTube Shorts Generator</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        brand: {
                            50: '#eef2ff',
                            100: '#e0e7ff',
                            400: '#818cf8',
                            500: '#6366f1',
                            600: '#4f46e5',
                            700: '#4338ca',
                        },
                        dark: {
                            900: '#0b0f19',
                            800: '#111827',
                            750: '#161f33',
                            700: '#1f2937',
                            600: '#374151'
                        }
                    },
                    fontFamily: {
                        sans: ['Plus Jakarta Sans', 'Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
                    }
                }
            }
        }
    </script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #0b0f19;
            color: #f3f4f6;
        }
        .gradient-text {
            background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .gradient-btn {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            box-shadow: 0 4px 20px -2px rgba(99, 102, 241, 0.45);
        }
        .gradient-btn:hover {
            box-shadow: 0 6px 25px rgba(99, 102, 241, 0.65);
            transform: translateY(-1px);
        }
        .glass-card {
            background: rgba(22, 31, 51, 0.7);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .glass-card:hover {
            border-color: rgba(99, 102, 241, 0.3);
        }
        .custom-scrollbar::-webkit-scrollbar {
            width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }
    </style>
</head>
<body class="min-h-screen custom-scrollbar flex flex-col">

    <!-- Top Navbar -->
    <header class="border-b border-white/10 bg-dark-900/80 backdrop-blur-md sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl gradient-btn flex items-center justify-center shadow-lg shadow-indigo-500/20">
                    <i data-lucide="video" class="w-5 h-5 text-white"></i>
                </div>
                <div>
                    <h1 class="font-extrabold text-lg tracking-tight gradient-text">Shorts AI</h1>
                    <p class="text-xs text-gray-400 font-medium">Viral 9:16 Generator</p>
                </div>
            </div>
            <div class="flex items-center space-x-3">
                <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    Ready
                </span>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        <!-- Hero Section -->
        <div class="text-center max-w-3xl mx-auto mb-10">
            <div class="inline-flex items-center gap-2 px-3.5 py-1 rounded-full text-xs font-bold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 mb-3 uppercase tracking-wider">
                <i data-lucide="sparkles" class="w-3.5 h-3.5"></i>
                AI-Powered Video Clipper
            </div>
            <h2 class="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight mb-3">
                Turn Long Videos into <span class="gradient-text">Viral Shorts</span>
            </h2>
            <p class="text-gray-400 text-sm sm:text-base leading-relaxed">
                Extract high-virality hooks with Whisper speech detection, auto-reframe into 9:16 with face tracking, and export ready-to-post vertical shorts.
            </p>
        </div>

        <!-- 2-Column Responsive Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            
            <!-- Left Column: Controls & Input (5 cols on lg) -->
            <div class="lg:col-span-5 space-y-6">
                
                <!-- Input Source Box -->
                <div class="glass-card rounded-2xl p-5 shadow-xl">
                    <h3 class="text-sm font-bold uppercase tracking-wider text-gray-400 mb-4 flex items-center gap-2">
                        <i data-lucide="link" class="w-4 h-4 text-indigo-400"></i>
                        1. Source Video
                    </h3>

                    <!-- Tabs -->
                    <div class="flex p-1 bg-dark-900/60 rounded-xl border border-white/5 mb-4">
                        <button id="tab-yt" onclick="switchTab('yt')" class="flex-1 py-2 px-3 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-2 bg-indigo-600 text-white shadow">
                            <i data-lucide="youtube" class="w-4 h-4 text-red-400"></i>
                            YouTube URL
                        </button>
                        <button id="tab-file" onclick="switchTab('file')" class="flex-1 py-2 px-3 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-2 text-gray-400 hover:text-white">
                            <i data-lucide="upload" class="w-4 h-4 text-indigo-400"></i>
                            Upload File
                        </button>
                    </div>

                    <!-- YouTube Input -->
                    <div id="panel-yt" class="space-y-3">
                        <div class="relative">
                            <input type="text" id="yt-url" placeholder="https://www.youtube.com/watch?v=..." oninput="handleUrlInput(this.value)" class="w-full bg-dark-900/80 border border-white/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 text-gray-100 placeholder-gray-500 pr-10">
                            <i data-lucide="search" class="w-4 h-4 text-gray-500 absolute right-3.5 top-3.5"></i>
                        </div>

                        <!-- Live YouTube Preview -->
                        <div id="yt-preview-container" class="rounded-xl overflow-hidden border border-white/10 bg-dark-900/80 aspect-video flex items-center justify-center">
                            <div class="text-center p-6 text-gray-500">
                                <i data-lucide="play-square" class="w-8 h-8 mx-auto mb-2 opacity-40"></i>
                                <p class="text-xs font-medium">Paste a YouTube URL to preview</p>
                            </div>
                        </div>
                    </div>

                    <!-- File Upload Input -->
                    <div id="panel-file" class="hidden space-y-3">
                        <div class="border-2 border-dashed border-white/10 hover:border-indigo-500/50 rounded-xl p-6 text-center cursor-pointer transition-colors bg-dark-900/40 relative">
                            <input type="file" id="video-file" accept="video/*" onchange="handleFileInput(event)" class="absolute inset-0 opacity-0 cursor-pointer w-full h-full">
                            <i data-lucide="cloud-upload" class="w-8 h-8 text-indigo-400 mx-auto mb-2"></i>
                            <p class="text-xs font-bold text-gray-200">Click to upload or drag & drop</p>
                            <p class="text-[11px] text-gray-500 mt-1">MP4, MOV, MKV up to 500MB</p>
                        </div>
                        <div id="file-name-display" class="hidden text-xs text-indigo-400 font-semibold px-2 flex items-center gap-1.5">
                            <i data-lucide="file-video" class="w-3.5 h-3.5"></i>
                            <span id="file-name-text">video.mp4</span>
                        </div>
                    </div>
                </div>

                <!-- Settings Box -->
                <div class="glass-card rounded-2xl p-5 shadow-xl space-y-4">
                    <h3 class="text-sm font-bold uppercase tracking-wider text-gray-400 flex items-center gap-2">
                        <i data-lucide="sliders" class="w-4 h-4 text-purple-400"></i>
                        2. Formatting & Mode
                    </h3>

                    <div class="grid grid-cols-2 gap-3">
                        <div>
                            <label class="block text-xs font-bold text-gray-300 mb-1.5">Framing Mode</label>
                            <select id="video-mode" class="w-full bg-dark-900/80 border border-white/10 rounded-xl px-3 py-2.5 text-xs font-medium focus:outline-none focus:border-indigo-500 text-gray-200">
                                <option value="crop-face" selected>🎯 Face Tracking (Auto)</option>
                                <option value="fit-center">⬛ Fit (Black Bars)</option>
                                <option value="fit-blur">✨ Fit (Blurred BG)</option>
                                <option value="crop-center">📐 Center Crop</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-xs font-bold text-gray-300 mb-1.5">Aspect Ratio</label>
                            <select id="aspect-ratio" class="w-full bg-dark-900/80 border border-white/10 rounded-xl px-3 py-2.5 text-xs font-medium focus:outline-none focus:border-indigo-500 text-gray-200">
                                <option value="9:16" selected>📱 9:16 (Shorts/TikTok)</option>
                                <option value="1:1">⏹️ 1:1 (Square)</option>
                                <option value="4:5">📸 4:5 (Social)</option>
                                <option value="16:9">🖥️ 16:9 (Landscape)</option>
                            </select>
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-3">
                        <div>
                            <label class="block text-xs font-bold text-gray-300 mb-1.5">Processing Engine</label>
                            <select id="engine-mode" class="w-full bg-dark-900/80 border border-white/10 rounded-xl px-3 py-2.5 text-xs font-medium focus:outline-none focus:border-indigo-500 text-gray-200">
                                <option value="local" selected>💻 Local (Whisper + FFmpeg)</option>
                                <option value="api">☁️ API (MuAPI Cloud)</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-xs font-bold text-gray-300 mb-1.5">Clips to Generate</label>
                            <select id="num-clips" class="w-full bg-dark-900/80 border border-white/10 rounded-xl px-3 py-2.5 text-xs font-medium focus:outline-none focus:border-indigo-500 text-gray-200">
                                <option value="1">1 Short</option>
                                <option value="2">2 Shorts</option>
                                <option value="3" selected>3 Shorts</option>
                                <option value="5">5 Shorts</option>
                            </select>
                        </div>
                    </div>

                    <!-- Custom Timestamps Toggle -->
                    <div class="pt-2 border-t border-white/5">
                        <label class="flex items-center gap-2 cursor-pointer">
                            <input type="checkbox" id="toggle-timestamps" onchange="document.getElementById('time-inputs').classList.toggle('hidden', !this.checked)" class="rounded bg-dark-900 border-white/20 text-indigo-600 focus:ring-indigo-500 w-4 h-4">
                            <span class="text-xs font-bold text-gray-300">Specify exact Start & End timestamps</span>
                        </label>
                        <div id="time-inputs" class="hidden grid grid-cols-2 gap-3 mt-3">
                            <div>
                                <label class="block text-[11px] font-semibold text-gray-400 mb-1">Start Time (sec)</label>
                                <input type="number" id="start-time" value="0" min="0" step="0.5" class="w-full bg-dark-900/80 border border-white/10 rounded-xl px-3 py-2 text-xs text-gray-200 focus:outline-none focus:border-indigo-500">
                            </div>
                            <div>
                                <label class="block text-[11px] font-semibold text-gray-400 mb-1">End Time (sec)</label>
                                <input type="number" id="end-time" value="60" min="0.1" step="0.5" class="w-full bg-dark-900/80 border border-white/10 rounded-xl px-3 py-2 text-xs text-gray-200 focus:outline-none focus:border-indigo-500">
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Generate Button -->
                <button id="generate-btn" onclick="startGeneration()" class="w-full gradient-btn text-white font-extrabold py-3.5 px-6 rounded-2xl text-base flex items-center justify-center gap-2 transition-all cursor-pointer">
                    <i data-lucide="sparkles" class="w-5 h-5"></i>
                    <span>Generate Shorts Now</span>
                </button>

            </div>

            <!-- Right Column: Rendered Shorts Showcase (7 cols on lg) -->
            <div class="lg:col-span-7 space-y-6">
                
                <div class="flex items-center justify-between">
                    <h3 class="text-lg font-bold flex items-center gap-2">
                        <i data-lucide="film" class="w-5 h-5 text-indigo-400"></i>
                        Generated Shorts Gallery
                    </h3>
                    <span id="results-count" class="text-xs text-gray-400 font-semibold">0 Clips Ready</span>
                </div>

                <!-- Loading State (Hidden by default) -->
                <div id="loading-state" class="hidden glass-card rounded-2xl p-8 text-center space-y-4">
                    <div class="w-14 h-14 rounded-full border-4 border-indigo-500/20 border-t-indigo-500 animate-spin mx-auto"></div>
                    <div>
                        <h4 id="loading-title" class="font-bold text-base text-gray-100">Processing Video...</h4>
                        <p id="loading-desc" class="text-xs text-gray-400 mt-1">Transcribing speech & analyzing virality hooks</p>
                    </div>
                    <div class="w-full bg-dark-900/80 rounded-full h-2 overflow-hidden max-w-xs mx-auto border border-white/5">
                        <div class="bg-gradient-to-r from-indigo-500 to-pink-500 h-full w-2/3 animate-pulse rounded-full"></div>
                    </div>
                </div>

                <!-- Empty State -->
                <div id="empty-state" class="glass-card rounded-2xl p-12 text-center border-dashed border-white/10">
                    <div class="w-16 h-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center mx-auto mb-4 text-indigo-400">
                        <i data-lucide="sparkles" class="w-8 h-8"></i>
                    </div>
                    <h4 class="font-bold text-gray-200 text-base mb-1">No Shorts Generated Yet</h4>
                    <p class="text-xs text-gray-500 max-w-sm mx-auto">
                        Paste a YouTube link or upload a file on the left, then click <strong>Generate Shorts</strong> to start.
                    </p>
                </div>

                <!-- Results Grid (9:16 Vertical Cards) -->
                <div id="results-grid" class="hidden grid grid-cols-1 sm:grid-cols-2 gap-5">
                    <!-- Dynamic Shorts Cards Injected Here -->
                </div>

            </div>

        </div>

    </main>

    <!-- Footer -->
    <footer class="mt-auto border-t border-white/10 py-6 text-center text-xs text-gray-500">
        <p>AI YouTube Shorts Generator • 100% Free & Open Source</p>
    </footer>

    <!-- Logic Script -->
    <script>
        lucide.createIcons();

        let activeTab = 'yt';

        function switchTab(tab) {
            activeTab = tab;
            const tabYt = document.getElementById('tab-yt');
            const tabFile = document.getElementById('tab-file');
            const panelYt = document.getElementById('panel-yt');
            const panelFile = document.getElementById('panel-file');

            if (tab === 'yt') {
                tabYt.className = 'flex-1 py-2 px-3 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-2 bg-indigo-600 text-white shadow';
                tabFile.className = 'flex-1 py-2 px-3 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-2 text-gray-400 hover:text-white';
                panelYt.classList.remove('hidden');
                panelFile.classList.add('hidden');
            } else {
                tabFile.className = 'flex-1 py-2 px-3 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-2 bg-indigo-600 text-white shadow';
                tabYt.className = 'flex-1 py-2 px-3 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-2 text-gray-400 hover:text-white';
                panelFile.classList.remove('hidden');
                panelYt.classList.add('hidden');
            }
        }

        function extractYtId(url) {
            if (!url) return null;
            const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=|shorts\/)([^#&?]*).*/;
            const match = url.match(regExp);
            return (match && match[2].length === 11) ? match[2] : null;
        }

        function handleUrlInput(url) {
            const container = document.getElementById('yt-preview-container');
            const vid = extractYtId(url);
            if (vid) {
                container.innerHTML = `<iframe class="w-full h-full" src="https://www.youtube.com/embed/${vid}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>`;
            } else {
                container.innerHTML = `
                    <div class="text-center p-6 text-gray-500">
                        <i data-lucide="play-square" class="w-8 h-8 mx-auto mb-2 opacity-40"></i>
                        <p class="text-xs font-medium">Paste a YouTube URL to preview</p>
                    </div>`;
                lucide.createIcons();
            }
        }

        function handleFileInput(e) {
            const file = e.target.files[0];
            const nameDisplay = document.getElementById('file-name-display');
            const nameText = document.getElementById('file-name-text');
            if (file) {
                nameText.innerText = file.name;
                nameDisplay.classList.remove('hidden');
            }
        }

        async function startGeneration() {
            const btn = document.getElementById('generate-btn');
            const loadingState = document.getElementById('loading-state');
            const emptyState = document.getElementById('empty-state');
            const resultsGrid = document.getElementById('results-grid');
            const resultsCount = document.getElementById('results-count');

            const formData = new FormData();

            if (activeTab === 'yt') {
                const url = document.getElementById('yt-url').value;
                if (!url) {
                    alert('Please enter a YouTube URL');
                    return;
                }
                formData.append('url', url);
            } else {
                const fileInput = document.getElementById('video-file');
                if (!fileInput.files[0]) {
                    alert('Please choose a video file');
                    return;
                }
                formData.append('file', fileInput.files[0]);
            }

            formData.append('mode', document.getElementById('engine-mode').value);
            formData.append('video_mode', document.getElementById('video-mode').value);
            formData.append('aspect_ratio', document.getElementById('aspect-ratio').value);
            formData.append('num_clips', document.getElementById('num-clips').value);

            const useTime = document.getElementById('toggle-timestamps').checked;
            formData.append('use_manual_time', useTime);
            if (useTime) {
                formData.append('start_time', document.getElementById('start-time').value);
                formData.append('end_time', document.getElementById('end-time').value);
            }

            // UI Loading state
            btn.disabled = true;
            btn.classList.add('opacity-50', 'cursor-not-allowed');
            emptyState.classList.add('hidden');
            resultsGrid.classList.add('hidden');
            loadingState.classList.remove('hidden');

            try {
                const res = await fetch('/api/generate', {
                    method: 'POST',
                    body: formData
                });

                const data = await res.json();
                if (!res.ok) {
                    throw new Error(data.detail || 'Failed to generate shorts');
                }

                renderResults(data.shorts || []);
            } catch (err) {
                alert('Error: ' + err.message);
                emptyState.classList.remove('hidden');
            } finally {
                loadingState.classList.add('hidden');
                btn.disabled = false;
                btn.classList.remove('opacity-50', 'cursor-not-allowed');
            }
        }

        function renderResults(shorts) {
            const resultsGrid = document.getElementById('results-grid');
            const resultsCount = document.getElementById('results-count');

            if (!shorts || shorts.length === 0) {
                document.getElementById('empty-state').classList.remove('hidden');
                resultsCount.innerText = '0 Clips';
                return;
            }

            resultsCount.innerText = `${shorts.length} Shorts Created`;
            resultsGrid.innerHTML = '';

            shorts.forEach(s => {
                const card = document.createElement('div');
                card.className = 'glass-card rounded-2xl p-4 flex flex-col justify-between space-y-3 transition-all hover:scale-[1.01] shadow-xl';

                const hookHtml = s.hook ? `
                    <div class="bg-indigo-500/10 border-l-2 border-indigo-400 p-2.5 rounded-r-lg text-xs italic text-indigo-200">
                        "${s.hook}"
                    </div>` : '';

                const videoPlayerHtml = s.clip_url ? `
                    <div class="relative w-full rounded-xl overflow-hidden bg-black aspect-[9/16] max-h-[420px] mx-auto shadow-lg border border-white/10">
                        <video src="${s.clip_url}" controls playsinline preload="metadata" class="w-full h-full object-cover"></video>
                    </div>` : `
                    <div class="aspect-[9/16] bg-dark-900/60 rounded-xl flex items-center justify-center text-xs text-red-400 p-4 border border-red-500/20 text-center">
                        Failed: ${s.error || 'Video rendering error'}
                    </div>`;

                const downloadBtnHtml = s.clip_url ? `
                    <a href="${s.clip_url}" download="short_${s.id}.mp4" class="w-full py-2.5 px-4 rounded-xl text-xs font-bold bg-white/10 hover:bg-white/20 text-white flex items-center justify-center gap-2 transition-colors border border-white/10">
                        <i data-lucide="download" class="w-4 h-4"></i>
                        Download Short #${s.id}
                    </a>` : '';

                card.innerHTML = `
                    <div class="space-y-2.5">
                        <div class="flex items-center justify-between">
                            <span class="px-2.5 py-0.5 rounded-full text-[11px] font-extrabold bg-gradient-to-r from-amber-500 to-red-500 text-white shadow-sm">
                                🔥 ${s.score}/100 VIRAL
                            </span>
                            <span class="text-[11px] font-bold text-indigo-300 bg-indigo-500/10 px-2 py-0.5 rounded-full border border-indigo-500/20">
                                ⏱️ ${s.duration}s (${s.start_time.toFixed(1)}s → ${s.end_time.toFixed(1)}s)
                            </span>
                        </div>
                        <h4 class="font-bold text-sm text-gray-100 leading-snug line-clamp-2">${s.title}</h4>
                        ${hookHtml}
                    </div>

                    ${videoPlayerHtml}
                    ${downloadBtnHtml}
                `;

                resultsGrid.appendChild(card);
            });

            resultsGrid.classList.remove('hidden');
            lucide.createIcons();
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 AI YouTube Shorts Generator Web UI Starting...")
    print("👉 Open your browser at: http://localhost:7860")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=7860)
