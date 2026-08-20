"""Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio. For 9:16 we slide a vertical
     window horizontally across the frame to keep faces centred (Haar
     cascade — same approach as the original repo, no external models).
"""
import os
import subprocess
from typing import Dict, List, Optional, Tuple

from ..config import LOCAL_OUTPUT_DIR


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -to end → re-encoded mp4 with audio."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", source_path,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _load_face_cascade(cv2):
    """Attempt to load Haar cascade for face detection."""
    try:
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades") and hasattr(cv2, "CascadeClassifier"):
            cascade_file = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(cascade_file):
                cc = cv2.CascadeClassifier(cascade_file)
                if not cc.empty():
                    return cc
        if hasattr(cv2, "CascadeClassifier"):
            cc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            if not cc.empty():
                return cc
    except Exception:
        pass
    return None


def _get_target_dims(aspect_ratio: str) -> Tuple[int, int]:
    """Calculate target canvas dimensions (w, h) in even numbers."""
    ratio = _ratio(aspect_ratio)
    if abs(ratio - 9.0 / 16.0) < 0.01:
        return 1080, 1920
    if abs(ratio - 1.0) < 0.01:
        return 1080, 1080
    if abs(ratio - 16.0 / 9.0) < 0.01:
        return 1920, 1080
    if ratio < 1.0:
        h = 1920
        w = int(h * ratio)
    else:
        w = 1920
        h = int(w / ratio)
    w = max(2, w - (w % 2))
    h = max(2, h - (h % 2))
    return w, h


def _reframe_crop_center(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Center crop video to target aspect ratio using ffmpeg."""
    ratio = _ratio(aspect_ratio)
    vf = f"crop='if(gt(a,{ratio}),ih*{ratio},iw)':'if(gt(a,{ratio}),ih,iw/{ratio})':'(in_w-out_w)/2':'(in_h-out_h)/2'"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _reframe_fit_center(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Fit entire video centered inside target aspect ratio with black padding."""
    target_w, target_h = _get_target_dims(aspect_ratio)
    vf = f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _reframe_fit_blur(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Fit entire video centered with a blurred, zoomed background filling the frame."""
    target_w, target_h = _get_target_dims(aspect_ratio)
    vf = (
        f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=25:5[bg];"
        f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_path,
        "-filter_complex", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _reframe_vertical_cv2(cv2, in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Crop the cut clip to the target aspect ratio using OpenCV face tracking."""
    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Compute the largest crop that fits inside the frame at the target ratio.
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    face_cascade = _load_face_cascade(cv2)

    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))

    last_center: Optional[Tuple[int, int]] = None
    smoothing = 0.15  # how aggressively to chase a new face position
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if face_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                if len(faces) > 0:
                    # Pick the largest face — usually the speaker.
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    cx = x + w // 2
                    cy = y + h // 2
                    if last_center is None:
                        last_center = (cx, cy)
                    else:
                        lx, ly = last_center
                        last_center = (
                            int(lx + (cx - lx) * smoothing),
                            int(ly + (cy - ly) * smoothing),
                        )
            except Exception:
                pass

        if last_center is None:
            last_center = (src_w // 2, src_h // 2)

        cx, cy = last_center
        x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
        y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        writer.write(cropped)

    cap.release()
    writer.release()

    # Mux audio from the cut clip back onto the silent reframed video.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    if os.path.exists(silent_path):
        os.remove(silent_path)
    return out_path


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str, video_mode: str = "crop-face") -> str:
    """Reframe video using the selected video mode."""
    mode = (video_mode or "crop-face").lower()
    if mode in ("fit-center", "fit", "pad", "letterbox"):
        return _reframe_fit_center(in_path, out_path, aspect_ratio)
    elif mode in ("fit-blur", "blur"):
        return _reframe_fit_blur(in_path, out_path, aspect_ratio)
    elif mode in ("crop-center", "center"):
        return _reframe_crop_center(in_path, out_path, aspect_ratio)
    else:  # "crop-face", "face", or default
        try:
            import cv2  # type: ignore
            return _reframe_vertical_cv2(cv2, in_path, out_path, aspect_ratio)
        except Exception as e:
            print(f"[clip/local] Notice: Face tracking unavailable ({e}). Falling back to ffmpeg center crop.", flush=True)
            return _reframe_crop_center(in_path, out_path, aspect_ratio)


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str = "9:16",
    out_path: str = "",
    video_mode: str = "crop-face",
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path."""
    cut_path = out_path + ".cut.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        _reframe_vertical(cut_path, out_path, aspect_ratio, video_mode=video_mode)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    video_mode: str = "crop-face",
) -> List[Dict]:
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"short_{i:02d}.mp4")
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')} [mode: {video_mode}]", flush=True)
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio=aspect_ratio,
                out_path=out_path,
                video_mode=video_mode,
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
