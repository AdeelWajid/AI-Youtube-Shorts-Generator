"""End-to-end orchestrator.

Two modes:
  * mode="api"   (default) — MuAPI does download / transcribe / LLM / autocrop.
                              Fast, no local deps, pay-per-call.
  * mode="local"            — yt-dlp + faster-whisper + OpenAI or Gemini + ffmpeg/opencv.
                              Self-hosted, LLM_PROVIDER selects OpenAI or Gemini.
"""
from typing import Dict, List, Optional, Tuple

from .clipper import crop_highlights
from .downloader import download_youtube
from .highlights import call_muapi_llm, get_highlights
from .transcriber import transcribe


def _extract_clip_info_from_transcript(
    transcript: Dict, start_time: float, end_time: float
) -> Tuple[str, str]:
    """Extract title and hook_sentence from transcript segments overlapping [start_time, end_time]."""
    segments = [
        seg for seg in transcript.get("segments", [])
        if float(seg.get("end", 0.0)) >= start_time and float(seg.get("start", 0.0)) <= end_time
    ]
    if not segments:
        return f"Custom Clip ({start_time:.1f}s - {end_time:.1f}s)", ""

    texts = [str(seg.get("text", "")).strip() for seg in segments if str(seg.get("text", "")).strip()]
    if not texts:
        return f"Custom Clip ({start_time:.1f}s - {end_time:.1f}s)", ""

    hook = texts[0]
    # If the first segment is very short (1-2 words), combine with the next segment for a better hook
    if len(texts) > 1 and len(hook.split()) <= 3:
        hook = f"{texts[0]} {texts[1]}"

    title = hook
    if len(title) > 60:
        title = title[:57].rstrip() + "..."

    return title, hook


def _run_local(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    video_mode: str = "crop-face",
) -> Dict:
    from .local.clipper import crop_highlights_local
    from .local.downloader import download_youtube_local
    from .local.llm import call_local_llm
    from .local.transcriber import transcribe_local

    source_path = download_youtube_local(youtube_url, fmt=download_format)

    if start_time is not None and end_time is not None:
        if end_time <= start_time:
            raise ValueError(f"end_time ({end_time}) must be greater than start_time ({start_time})")

        # Load or generate transcript to populate hook & title
        try:
            transcript = transcribe_local(source_path, language=language)
            title, hook = _extract_clip_info_from_transcript(transcript, start_time, end_time)
        except Exception as e:
            print(f"[pipeline/local] Note: Could not extract transcript for hook ({e})", flush=True)
            transcript = {"segments": []}
            title, hook = f"Custom Clip ({start_time:.1f}s - {end_time:.1f}s)", ""

        custom_highlight = {
            "start_time": float(start_time),
            "end_time": float(end_time),
            "title": title,
            "hook_sentence": hook,
            "score": 100,
        }
        shorts = crop_highlights_local(
            source_path, [custom_highlight], aspect_ratio=aspect_ratio, video_mode=video_mode
        )
        return {
            "mode": "local",
            "source_video_url": source_path,
            "transcript": transcript,
            "highlights": [custom_highlight],
            "shorts": shorts,
        }

    transcript = transcribe_local(source_path, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    if start_time is not None or end_time is not None:
        filtered_segments = []
        for seg in transcript["segments"]:
            s_start = seg.get("start", 0.0)
            s_end = seg.get("end", 0.0)
            if start_time is not None and s_end < start_time:
                continue
            if end_time is not None and s_start > end_time:
                continue
            filtered_segments.append(seg)
        if not filtered_segments:
            raise RuntimeError(
                f"No transcript segments found in range {start_time or 0}s - {end_time or 'end'}s."
            )
        transcript = {**transcript, "segments": filtered_segments}

    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_local_llm)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
    print(f"[pipeline/local] cropping {len(top)} of {len(all_highlights)} candidates [mode: {video_mode}]", flush=True)

    shorts = crop_highlights_local(source_path, top, aspect_ratio=aspect_ratio, video_mode=video_mode)

    return {
        "mode": "local",
        "source_video_url": source_path,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def _run_api(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    video_mode: str = "crop-face",
) -> Dict:
    source_url = download_youtube(youtube_url, fmt=download_format)

    if start_time is not None and end_time is not None:
        if end_time <= start_time:
            raise ValueError(f"end_time ({end_time}) must be greater than start_time ({start_time})")

        try:
            transcript = transcribe(source_url, language=language)
            title, hook = _extract_clip_info_from_transcript(transcript, start_time, end_time)
        except Exception as e:
            print(f"[pipeline] Note: Could not extract transcript for hook ({e})", flush=True)
            transcript = {"segments": []}
            title, hook = f"Custom Clip ({start_time:.1f}s - {end_time:.1f}s)", ""

        custom_highlight = {
            "start_time": float(start_time),
            "end_time": float(end_time),
            "title": title,
            "hook_sentence": hook,
            "score": 100,
        }
        shorts = crop_highlights(source_url, [custom_highlight], aspect_ratio=aspect_ratio)
        return {
            "mode": "api",
            "source_video_url": source_url,
            "transcript": transcript,
            "highlights": [custom_highlight],
            "shorts": shorts,
        }

    transcript = transcribe(source_url, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    if start_time is not None or end_time is not None:
        filtered_segments = []
        for seg in transcript["segments"]:
            s_start = seg.get("start", 0.0)
            s_end = seg.get("end", 0.0)
            if start_time is not None and s_end < start_time:
                continue
            if end_time is not None and s_start > end_time:
                continue
            filtered_segments.append(seg)
        if not filtered_segments:
            raise RuntimeError(
                f"No transcript segments found in range {start_time or 0}s - {end_time or 'end'}s."
            )
        transcript = {**transcript, "segments": filtered_segments}

    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_muapi_llm)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
    print(f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    shorts = crop_highlights(source_url, top, aspect_ratio=aspect_ratio)

    return {
        "mode": "api",
        "source_video_url": source_url,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def generate_shorts(
    youtube_url: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
    mode: str = "api",
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    video_mode: str = "crop-face",
) -> Dict:
    """Run the full pipeline and return a structured result.

    Args:
        youtube_url: source URL.
        num_clips: how many shorts to render.
        aspect_ratio: e.g. "9:16", "1:1".
        download_format: source resolution ("360" / "480" / "720" / "1080").
        language: ISO-639-1 to force Whisper language detection.
        mode: "api" (default, MuAPI) or "local" (yt-dlp + faster-whisper +
            OpenAI or Gemini + ffmpeg).
        start_time: optional start time in seconds (e.g. 194.0).
        end_time: optional end time in seconds (e.g. 240.0).
        video_mode: framing/crop style ("crop-face", "fit-center", "fit-blur", "crop-center").

    Returns:
        {
          "mode": "api" | "local",
          "source_video_url": str,   # hosted URL (api) or local path (local)
          "transcript": {...},
          "highlights": [...],       # all candidates ranked
          "shorts": [...],           # top `num_clips` with clip_url / local path
        }
    """
    mode = (mode or "api").lower()
    if mode == "local":
        return _run_local(
            youtube_url,
            num_clips,
            aspect_ratio,
            download_format,
            language,
            start_time,
            end_time,
            video_mode=video_mode,
        )
    if mode == "api":
        return _run_api(
            youtube_url,
            num_clips,
            aspect_ratio,
            download_format,
            language,
            start_time,
            end_time,
            video_mode=video_mode,
        )
    raise ValueError(f"Unknown mode: {mode!r}. Use 'api' or 'local'.")
