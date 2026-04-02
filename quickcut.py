"""QuickCut — Standalone 2-pass clip enhancer.

Drop in a clip, get enhanced output with captions, hook text, B-roll,
dynamic edits, and montage — all in just 2 FFmpeg passes.

Usage: python quickcut.py --input clip.mp4 --broll-source ai
"""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

import anthropic
import httpx
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REPLICATE_HOSTS = {"replicate.delivery", "pbxt.replicate.delivery"}
_MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50 MB
_VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "../video-forge-lite/config.yaml") -> dict:
    """Load YAML config with ${ENV_VAR} substitution."""
    raw = Path(config_path).read_text(encoding="utf-8")
    resolved = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), raw)
    return yaml.safe_load(resolved)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_duration(path: str | Path) -> float:
    """Get video duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _ass_filter_path(p: str) -> str:
    """Convert a path to FFmpeg ASS filter format (Windows-safe)."""
    return str(p).replace("\\", "/").replace(":", "\\:")


def _run_ffmpeg(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run an FFmpeg/ffprobe command, surfacing stderr on failure."""
    try:
        return subprocess.run(cmd, capture_output=True, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode(errors="replace").strip()
        if stderr:
            print(f"[ffmpeg] stderr:\n{stderr[-500:]}")
        raise


def _safe_int(value, default: int, lo: int = 1, hi: int = 10000) -> int:
    """Coerce a config value to int within bounds."""
    try:
        v = int(value)
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float, lo: float = 0.0, hi: float = 1000.0) -> float:
    """Coerce a config value to float within bounds."""
    try:
        v = float(value)
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return default


_PEXELS_DOWNLOAD_HOSTS = {"www.pexels.com", "videos.pexels.com", "video-previews.pexels.com"}


def _validate_download_url(url: str, allowed_hosts: set[str]) -> str:
    """Validate a download URL is HTTPS and from an allowed host."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise RuntimeError(f"Untrusted URL scheme: {url!r}")
    host = parsed.hostname or ""
    if not any(host == h or host.endswith(f".{h}") for h in allowed_hosts):
        raise RuntimeError(f"Untrusted download host: {host!r}")
    return url


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(video_path: str, config: dict) -> dict:
    """Transcribe a video using faster-whisper with word timestamps."""
    from faster_whisper import WhisperModel

    whisper_cfg = config.get("whisper", {})
    model_size = whisper_cfg.get("model", "base")
    language = whisper_cfg.get("language", "en")

    # Extract audio to temp WAV (use tempfile to avoid predictable paths)
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="quickcut_audio_")
    os.close(wav_fd)
    _run_ffmpeg(["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", "-y", wav_path])

    try:
        # Device detection
        device = whisper_cfg.get("device", "auto")
        compute_type = whisper_cfg.get("compute_type", "float16")
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if device == "cpu":
            compute_type = "int8"

        print(f"[transcribe] Loading whisper model: {model_size} ({device}/{compute_type})")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        print("[transcribe] Transcribing...")
        segments_iter, info = model.transcribe(wav_path, language=language, word_timestamps=True)

        segments = []
        full_text_parts = []
        for seg in segments_iter:
            words = [
                {"word": w.word.strip(), "start": round(w.start, 3),
                 "end": round(w.end, 3), "probability": round(w.probability, 3)}
                for w in (seg.words or [])
            ]
            segments.append({
                "id": seg.id, "start": round(seg.start, 3),
                "end": round(seg.end, 3), "text": seg.text.strip(), "words": words,
            })
            full_text_parts.append(seg.text.strip())

        transcript = {
            "language": info.language, "duration": round(info.duration, 2),
            "full_text": " ".join(full_text_parts), "segments": segments,
        }
        print(f"[transcribe] Done — {transcript['duration']}s, {len(segments)} segments")
        return transcript
    finally:
        Path(wav_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# B-roll analysis via Claude
# ---------------------------------------------------------------------------

def analyze_broll(transcript: dict, config: dict) -> list[dict]:
    """Ask Claude to identify B-roll overlay moments."""
    duration = transcript["duration"]
    broll_cfg = config.get("broll", {})
    max_clips = max(1, int(duration / 10) - 1)
    max_clips = min(max_clips, broll_cfg.get("max_clips_per_video", 3))
    clip_dur = broll_cfg.get("clip_duration", 3.5)
    hook_prot = config.get("dynamic_edits", {}).get("hook_protection", 5.0)

    lines = []
    for seg in transcript["segments"]:
        m, s = int(seg["start"] // 60), seg["start"] % 60
        lines.append(f"[{m}:{s:05.2f}] {seg['text']}")

    prompt = (
        f"Analyze this transcript and identify up to {max_clips} moments for B-roll overlay.\n\n"
        f"Rules:\n"
        f"- No B-roll in the first {hook_prot} seconds (hook protection — speaker's face must show)\n"
        f"- Each B-roll clip is {clip_dur} seconds long\n"
        f"- Space clips roughly every 10 seconds\n"
        f"- Use LITERAL keywords matching what the speaker says "
        f"(if they say \"money\", use \"money\" not \"wealth\")\n"
        f"- Skip punchlines and emotional delivery moments (speaker's face matters there)\n"
        f"- No overlapping clips\n\n"
        f"Transcript:\n{chr(10).join(lines)}\n\n"
        f'Return ONLY a JSON array: [{{"start_time": float, "end_time": float, '
        f'"keyword": "string", "reason": "string"}}]'
    )

    api_key = config.get("api_keys", {}).get("anthropic", "")
    model = config.get("claude", {}).get("model", "claude-sonnet-4-20250514")

    print(f"[broll] Analyzing transcript for B-roll moments (max {max_clips})...")
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model, max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            moments = json.loads(match.group())[:max_clips]
        except json.JSONDecodeError as e:
            print(f"[broll] Failed to parse Claude response as JSON: {e}")
            return []
        # Validate timing values are within video duration
        validated = []
        for m in moments:
            try:
                st = float(m.get("start_time", -1))
                et = float(m.get("end_time", -1))
                if 0 <= st < et <= duration + 1 and isinstance(m.get("keyword"), str):
                    m["start_time"], m["end_time"] = st, et
                    validated.append(m)
            except (TypeError, ValueError):
                continue
        print(f"[broll] Found {len(validated)} B-roll moments")
        return validated
    print("[broll] No B-roll moments found")
    return []


# ---------------------------------------------------------------------------
# B-roll fetching — Pexels
# ---------------------------------------------------------------------------

def fetch_pexels(keyword: str, config: dict) -> str:
    """Download a portrait B-roll clip from Pexels."""
    broll_cfg = config.get("broll", {})
    api_key = broll_cfg.get("pexels_api_key", "")
    cache_dir = broll_cfg.get("cache_dir", "./broll_cache")

    slug = keyword.lower().replace(" ", "_")[:50]
    cache_path = Path(cache_dir) / f"{slug}.mp4"
    if cache_path.exists():
        return str(cache_path)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    resp = httpx.get(
        "https://api.pexels.com/videos/search",
        params={"query": keyword, "per_page": 3, "orientation": "portrait"},
        headers={"Authorization": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    videos = resp.json().get("videos", [])
    if not videos:
        raise RuntimeError(f"No Pexels results for '{keyword}'")

    url = None
    for video in videos:
        files = video.get("video_files", [])
        mp4s = [f for f in files if f.get("file_type") == "video/mp4"]
        if not mp4s:
            continue
        hd_portrait = [f for f in mp4s if f.get("quality") == "hd"
                       and f.get("height", 0) > f.get("width", 0)]
        hd_any = [f for f in mp4s if f.get("quality") == "hd"]
        pick = (hd_portrait or hd_any or mp4s)[0]
        url = pick["link"]
        break

    if not url:
        raise RuntimeError(f"No video files found for '{keyword}'")

    # Validate Pexels download URL (SSRF protection)
    _validate_download_url(url, _PEXELS_DOWNLOAD_HOSTS)

    print(f"[broll] Downloading Pexels clip: {keyword}")
    with httpx.stream("GET", url, timeout=60) as stream:
        stream.raise_for_status()
        with open(cache_path, "wb") as f:
            for chunk in stream.iter_bytes(8192):
                f.write(chunk)
    return str(cache_path)


# ---------------------------------------------------------------------------
# B-roll fetching — AI (Replicate Flux + Ken Burns)
# ---------------------------------------------------------------------------

def _validate_replicate_url(url: str) -> str:
    """Ensure the URL is an HTTPS Replicate delivery URL (SSRF protection)."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise RuntimeError(f"Untrusted URL scheme from Replicate: {url!r}")
    host = parsed.hostname or ""
    if not any(host == h or host.endswith(f".{h}") for h in _REPLICATE_HOSTS):
        raise RuntimeError(f"Untrusted host from Replicate: {host!r}")
    return url


def fetch_ai_broll(keyword: str, reason: str, duration: float, config: dict) -> str:
    """Generate an AI image via Flux and convert to Ken Burns video."""
    broll_cfg = config.get("broll", {})
    cache_dir = broll_cfg.get("ai_cache_dir", "./broll_cache/ai")
    api_token = broll_cfg.get("replicate_api_token", "")
    if not api_token:
        raise RuntimeError("REPLICATE_API_TOKEN not set in config/env")

    slug = re.sub(r"[^\w\-]", "", keyword.lower().replace(" ", "_")[:50]) or "broll"
    video_path = Path(cache_dir) / f"{slug}_video.mp4"
    if video_path.exists():
        return str(video_path)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Generate image prompt via Claude
    image_prompt_text = (
        f'You are a visual director creating a single image prompt for AI image generation (Flux).\n\n'
        f'CONTEXT:\n- B-roll keyword: "{keyword}"\n- Why this image is needed: "{reason}"\n'
        f'- The image will overlay a talking-head video for 3-4 seconds\n'
        f'- Output format: 9:16 vertical (portrait) — like a phone screen\n\n'
        f'Write a single, detailed image prompt (1-3 sentences) that would generate a cinematic, '
        f'professional photo. Follow these rules:\n\n'
        f'1. Describe a SPECIFIC scene featuring a SAMURAI IN FULL TRADITIONAL ARMOR (yoroi) '
        f'performing the action or interacting with the concept described by the keyword. '
        f'The samurai should be the focal point — shown from behind, from the side, or in a '
        f'wide shot (never a close-up face).\n'
        f'2. Include lighting direction (e.g. "warm golden hour light from left")\n'
        f'3. Include camera angle or lens (e.g. "shot from behind", "wide cinematic 35mm")\n'
        f'4. Use a vibrant, premium aesthetic — think: cinematic still from a samurai film\n'
        f'5. The samurai should be DOING the thing the keyword describes\n'
        f'6. NO text, logos, watermarks, or UI elements\n'
        f'7. Use BRIGHT, well-lit scenes with high contrast and vivid color\n\n'
        f'Return ONLY the image prompt text. No quotes, no explanation, no formatting.'
    )

    api_key = config.get("api_keys", {}).get("anthropic", "")
    model = config.get("claude", {}).get("model", "claude-sonnet-4-20250514")
    client = anthropic.Anthropic(api_key=api_key)

    print(f"[ai-broll] Generating image prompt for: {keyword}")
    prompt_resp = client.messages.create(
        model=model, max_tokens=300,
        messages=[{"role": "user", "content": image_prompt_text}],
    )
    flux_prompt = prompt_resp.content[0].text.strip()

    # Call Replicate Flux
    print(f"[ai-broll] Calling Flux: {flux_prompt[:80]}...")
    resp = httpx.post(
        "https://api.replicate.com/v1/models/black-forest-labs/flux-1.1-pro/predictions",
        headers={"Authorization": f"Bearer {api_token}",
                 "Content-Type": "application/json", "Prefer": "wait"},
        json={"input": {"prompt": flux_prompt, "aspect_ratio": "9:16",
                        "output_format": "png", "output_quality": 90,
                        "safety_tolerance": 5}},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    output_url = data.get("output")
    if isinstance(output_url, list):
        output_url = output_url[0] if output_url else None
    if not output_url:
        raise RuntimeError(f"Flux generation failed: {data.get('error', 'Unknown')}")

    output_url = _validate_replicate_url(output_url)

    # Download image with validation
    img_resp = httpx.get(output_url, timeout=60)
    img_resp.raise_for_status()
    content_type = img_resp.headers.get("content-type", "")
    if "image" not in content_type:
        raise RuntimeError(f"Unexpected content-type from Replicate: {content_type}")
    if len(img_resp.content) > _MAX_IMAGE_BYTES:
        raise RuntimeError("Replicate image exceeds 50 MB size limit")

    image_path = Path(cache_dir) / f"{slug}.png"
    image_path.write_bytes(img_resp.content)
    print(f"[ai-broll] Image saved: {image_path}")

    # Ken Burns video
    res = config.get("rendering", {}).get("output_resolution", [1080, 1920])
    w, h = res[0], res[1]
    fps = 30
    total_frames = int(duration * fps)
    fade_dur = min(0.3, duration / 3)

    _run_ffmpeg([
        "ffmpeg", "-loop", "1", "-i", str(image_path),
        "-vf", (
            f"scale={int(w * 1.3)}:{int(h * 1.3)},"
            f"zoompan=z='1.09-0.09*on/{total_frames}':"
            f"x='iw/2-(iw/zoom/2)-(iw*0.012)*(1-on/{total_frames})':"
            f"y='ih/2-(ih/zoom/2)-(ih*0.012)*(1-on/{total_frames})':"
            f"d={total_frames}:s={w}x{h}:fps={fps},"
            f"fade=t=in:st=0:d={fade_dur},"
            f"fade=t=out:st={max(0.0, duration - fade_dur)}:d={fade_dur}"
        ),
        "-c:v", "libx264", "-t", str(duration),
        "-pix_fmt", "yuv420p", "-y", str(video_path),
    ])

    print(f"[ai-broll] Ken Burns video: {video_path}")
    return str(video_path)


# ---------------------------------------------------------------------------
# Dynamic edit planning
# ---------------------------------------------------------------------------

def plan_edits(duration: float, broll_segments: list[dict], config: dict) -> list[dict]:
    """Generate zoom/shift edit points, avoiding hook and B-roll regions."""
    cfg = config.get("dynamic_edits", {})
    interval = cfg.get("interval", 4.0)
    zoom_dur = cfg.get("zoom_duration", 1.5)
    hook_protection = cfg.get("hook_protection", 5.0)

    broll_ranges = [(s["start_time"], s["end_time"]) for s in broll_segments]
    points, t, cycle_idx = [], interval, 0
    cycle = ["zoom_in", "zoom_out", "shift"]

    while t < duration:
        if t < hook_protection:
            t += interval
            continue
        if not any(start <= t <= end for start, end in broll_ranges):
            points.append({"time": round(t, 2), "duration": zoom_dur,
                           "type": cycle[cycle_idx % 3]})
            cycle_idx += 1
        t += interval
    return points


# ---------------------------------------------------------------------------
# Montage builder
# ---------------------------------------------------------------------------

def build_montage(config: dict, tmp_dir: str) -> tuple[str, float]:
    """Build a randomized lifestyle montage from B-roll clips."""
    cfg = config.get("montage", {})
    broll_dir = Path(cfg.get("broll_dir", ""))

    if not broll_dir.exists():
        print(f"[montage] B-roll dir not found: {broll_dir}")
        return "", 0.0

    clips = [f for f in broll_dir.rglob("*") if f.suffix.lower() in _VIDEO_EXTS]
    if not clips:
        print("[montage] No video files found")
        return "", 0.0

    motion_clips = [c for c in clips if "static" not in c.stem.lower()]
    pool = motion_clips if len(motion_clips) >= 3 else clips

    count_range = cfg.get("clip_count", [3, 3])
    dur_range = cfg.get("clip_duration", [1.2, 1.2])
    res = config.get("rendering", {}).get("output_resolution", [1080, 1920])
    w, h = res[0], res[1]
    fps = config.get("rendering", {}).get("fps", 30)

    count = random.randint(count_range[0], count_range[1])
    selected = random.sample(pool, min(count, len(pool)))

    print(f"[montage] Building from {len(selected)} clips...")
    segments = []
    for i, clip in enumerate(selected):
        clip_dur = _get_duration(clip)
        if clip_dur < 2.0:
            continue
        seg_len = random.uniform(dur_range[0], dur_range[1])
        max_start = max(0.0, clip_dur - seg_len - 1.0)
        start = random.uniform(0.0, max_start) if max_start > 0 else 0.0

        seg_path = str(Path(tmp_dir) / f"seg_{i}.mp4")
        try:
            _run_ffmpeg([
                "ffmpeg", "-ss", f"{start:.2f}", "-i", str(clip),
                "-t", f"{seg_len:.2f}",
                "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                       f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black",
                "-r", str(fps), "-c:v", "libx264", "-crf", "18",
                "-pix_fmt", "yuv420p", "-an", "-y", seg_path,
            ])
            segments.append(seg_path)
        except subprocess.CalledProcessError:
            print(f"[montage] Failed to extract from {clip.name}, skipping")

    if not segments:
        return "", 0.0

    # Concat via demuxer
    list_path = str(Path(tmp_dir) / f"concat_{uuid.uuid4().hex[:8]}.txt")
    with open(list_path, "w") as f:
        for p in segments:
            f.write(f"file '{str(p).replace(chr(92), '/')}'\n")

    montage_path = str(Path(tmp_dir) / "montage.mp4")
    _run_ffmpeg(["ffmpeg", "-f", "concat", "-safe", "0", "-i", list_path,
                 "-c", "copy", "-y", montage_path])
    montage_dur = _get_duration(montage_path)
    print(f"[montage] Built: {montage_dur:.2f}s from {len(segments)} clips")
    return montage_path, montage_dur


# ---------------------------------------------------------------------------
# ASS subtitle generation
# ---------------------------------------------------------------------------

def hex_to_ass_color(hex_color: str) -> str:
    """Convert #RRGGBB to ASS &H00BBGGRR& format."""
    h = hex_color.lstrip("#")
    return f"&H00{h[4:6].upper()}{h[2:4].upper()}{h[0:2].upper()}&"


def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS time H:MM:SS.cc."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    centis = int((secs % 1) * 100)
    return f"{hours}:{minutes:02d}:{int(secs):02d}.{centis:02d}"


def _get_caption_config(config: dict, platform: str | None = None) -> dict:
    """Resolve caption config for a platform, falling back to default."""
    captions = config.get("captions", {})
    default = captions.get("default", captions)
    if platform and platform in captions:
        return {**default, **captions[platform]}
    return default


def build_caption_ass(transcript: dict, output_path: str, config: dict,
                      platform: str | None = None) -> str:
    """Create ASS subtitle file with word-by-word highlight captions."""
    cap_cfg = _get_caption_config(config, platform)
    res = config.get("rendering", {}).get("output_resolution", [1080, 1920])
    font = cap_cfg.get("font", "Montserrat-Bold")
    font_size = cap_cfg.get("font_size", 60)
    inactive = hex_to_ass_color(cap_cfg.get("inactive_color", "#FFFFFF"))
    active = hex_to_ass_color(cap_cfg.get("active_color", "#FFFF00"))
    shadow = hex_to_ass_color(cap_cfg.get("shadow_color", "#000000"))
    max_visible = cap_cfg.get("max_words_visible", 3)
    position = cap_cfg.get("position", "lower_quarter")
    alignment = 2
    margin_v = int(res[1] * 0.15) if position == "lower_quarter" else 80

    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {res[0]}\nPlayResY: {res[1]}\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{font_size},{inactive},{inactive},"
        f"{shadow},{shadow},-1,0,0,0,100,100,0,0,1,3,1,"
        f"{alignment},10,10,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    # Flatten all words
    words = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            words.append({"word": w["word"], "start": w["start"], "end": w["end"]})

    events = []
    for i, word in enumerate(words):
        start = _format_ass_time(word["start"])
        end = _format_ass_time(word["end"])
        half = max_visible // 2
        win_start = max(0, i - half)
        win_end = min(len(words), win_start + max_visible)
        if win_end - win_start < max_visible:
            win_start = max(0, win_end - max_visible)

        parts = []
        for j in range(win_start, win_end):
            w = words[j]["word"]
            if j == i:
                parts.append(f"{{\\c{active}\\fscx110\\fscy110}}{w}"
                             f"{{\\c{inactive}\\fscx100\\fscy100}}")
            else:
                parts.append(w)
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{' '.join(parts)}")

    Path(output_path).write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    print(f"[captions] Saved: {output_path}")
    return output_path


def extract_hook(transcript: dict, max_words: int = 7) -> dict | None:
    """Extract the opening hook sentence from the transcript."""
    segments = transcript.get("segments", [])
    if not segments:
        return None

    chosen = None
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        if len(text.split()) < 4 and text[0].islower():
            continue
        chosen = seg
        break
    if chosen is None:
        chosen = segments[0]

    text = chosen.get("text", "").strip()
    if not text:
        return None
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    if not text.endswith((".", "!", "?")):
        text = text.rstrip(",;:\u2014-") + "..."

    return {"text": text, "start": 0.0, "end": chosen.get("end", 4.0)}


def _get_hook_config(config: dict, platform: str | None = None) -> dict:
    """Resolve hook_text config for a platform, falling back to default."""
    ht = config.get("hook_text", {})
    default = ht.get("default", {})
    if platform and platform in ht:
        return {**default, **ht[platform]}
    return default


def build_hook_ass(hook: dict, output_path: str, config: dict,
                   platform: str | None = None) -> str:
    """Create ASS file for hook text overlay with pop-in animation."""
    hook_cfg = _get_hook_config(config, platform)
    res = config.get("rendering", {}).get("output_resolution", [1080, 1920])
    font = hook_cfg.get("font", "Montserrat-Bold")
    font_size = hook_cfg.get("font_size", 90)
    color = hex_to_ass_color(hook_cfg.get("color", "#FFFFFF"))
    shadow_color = hex_to_ass_color(hook_cfg.get("shadow_color", "#0D0D0D"))
    border_size = hook_cfg.get("border_size", 4)
    position = hook_cfg.get("position", "center")

    if position == "top":
        alignment, margin_v = 8, int(res[1] * 0.15)
    elif position == "center":
        alignment, margin_v = 5, 50
    else:
        pct = int(position) if str(position).isdigit() else 60
        alignment, margin_v = 2, int(res[1] * pct / 100)

    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {res[0]}\nPlayResY: {res[1]}\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: HookText,{font},{font_size},{color},{color},"
        f"{shadow_color},{shadow_color},-1,0,0,0,100,100,0,0,1,{border_size},2,"
        f"{alignment},40,40,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    duration = hook_cfg.get("duration", 4.0)
    pop_in = hook_cfg.get("pop_in_ms", 150)
    settle = pop_in + hook_cfg.get("settle_ms", 100)
    fade_out_ms = hook_cfg.get("fade_out_ms", 400)
    duration_ms = int(duration * 1000)
    fade_start = max(settle, duration_ms - fade_out_ms)

    start = _format_ass_time(0.0)
    end = _format_ass_time(duration)
    safe_text = hook["text"].replace("{", "").replace("}", "")

    anim = (
        f"{{\\fscx0\\fscy0\\alpha&HFF&"
        f"\\t(0,{pop_in},\\fscx130\\fscy130\\alpha&H00&)"
        f"\\t({pop_in},{settle},\\fscx100\\fscy100)"
        f"\\t({fade_start},{duration_ms},\\alpha&HFF&)}}"
    )
    event = f"Dialogue: 10,{start},{end},HookText,,0,0,0,,{anim}{safe_text}"

    Path(output_path).write_text(header + event + "\n", encoding="utf-8")
    print(f"[hook] Saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 2-Pass FFmpeg render
# ---------------------------------------------------------------------------

def render_enhanced(video_path: str, broll_clips: list[dict], edit_points: list[dict],
                    montage_path: str, montage_dur: float,
                    caption_ass: str, hook_ass: str,
                    output_path: str, config: dict, tmp_dir: str) -> str:
    """Render enhanced video in exactly 2 FFmpeg passes."""
    res = config.get("rendering", {}).get("output_resolution", [1080, 1920])
    w = _safe_int(res[0], 1080, 100, 7680)
    h = _safe_int(res[1], 1920, 100, 7680)
    fps = _safe_int(config.get("rendering", {}).get("fps", 30), 30, 1, 120)
    crf = _safe_int(config.get("rendering", {}).get("crf", 18), 18, 0, 51)
    cfg_edits = config.get("dynamic_edits", {})

    has_edits = bool(edit_points)
    has_montage = bool(montage_path)
    has_broll = bool(broll_clips)
    has_subs = bool(caption_ass) or bool(hook_ass)

    # If nothing to do in pass 1, go straight to pass 2
    if not has_edits and not has_montage and not has_broll:
        pass1_out = video_path
    else:
        # --- Build filter_complex for Pass 1 ---
        inputs = ["-i", video_path]
        filter_parts = []
        input_idx = 1

        if has_edits:
            zoom = _safe_float(cfg_edits.get("zoom_amount", 1.08), 1.08, 1.0, 2.0)
            base_zoom = 0.96
            shift_px = int(w * _safe_float(cfg_edits.get("shift_amount", 0.03), 0.03, 0.0, 0.2))

            z_parts, x_parts = [], []
            for pt in edit_points:
                f_start = int(pt["time"] * fps)
                f_end = int((pt["time"] + pt["duration"]) * fps)
                if pt["type"] == "zoom_in":
                    z_parts.append(f"if(between(on,{f_start},{f_end}),{zoom},")
                    x_parts.append(f"if(between(on,{f_start},{f_end}),iw/2-(iw/{zoom})/2,")
                elif pt["type"] == "zoom_out":
                    z_parts.append(f"if(between(on,{f_start},{f_end}),{base_zoom},")
                    x_parts.append(f"if(between(on,{f_start},{f_end}),iw/2-(iw/{base_zoom})/2,")
                elif pt["type"] == "shift":
                    z_parts.append(f"if(between(on,{f_start},{f_end}),1.0,")
                    x_parts.append(f"if(between(on,{f_start},{f_end}),iw/2-(iw/1.0)/2+{shift_px},")

            n = len(z_parts)
            z_expr = "".join(z_parts) + str(base_zoom) + ")" * n
            x_expr = "".join(x_parts) + f"iw/2-(iw/{base_zoom})/2" + ")" * n

            filter_parts.append(
                f"[0:v]zoompan=z='{z_expr}':x='{x_expr}'"
                f":y='ih/2-(ih/zoom)/2':d=1:s={w}x{h}:fps={fps}[edited]"
            )
        else:
            filter_parts.append(f"[0:v]scale={w}:{h}[edited]")

        prev_label = "edited"
        overlay_idx = 0

        if has_montage:
            inputs.extend(["-i", montage_path])
            m_idx = input_idx
            input_idx += 1
            out_label = f"v{overlay_idx}"
            filter_parts.append(
                f"[{m_idx}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setpts=PTS-STARTPTS[m]"
            )
            filter_parts.append(
                f"[{prev_label}][m]overlay=0:0:"
                f"enable='between(t,0,{montage_dur:.2f})'[{out_label}]"
            )
            prev_label = out_label
            overlay_idx += 1

        for seg in broll_clips:
            inputs.extend(["-i", seg["clip_path"]])
            b_idx = input_idx
            input_idx += 1
            b_label = f"b{b_idx}"
            out_label = f"v{overlay_idx}"
            start, end = seg["start_time"], seg["end_time"]
            filter_parts.append(
                f"[{b_idx}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
                f"setpts=PTS-STARTPTS[{b_label}]"
            )
            filter_parts.append(
                f"[{prev_label}][{b_label}]overlay=0:0:"
                f"enable='between(t,{start:.2f},{end:.2f})'[{out_label}]"
            )
            prev_label = out_label
            overlay_idx += 1

        pass1_out = str(Path(tmp_dir) / "pass1.mp4")
        filter_str = ";".join(filter_parts)

        print("[render] Pass 1: filter_complex (edits + overlays)...")
        cmd1 = ["ffmpeg"] + inputs + [
            "-filter_complex", filter_str,
            "-map", f"[{prev_label}]", "-map", "0:a?",
            "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-y", pass1_out,
        ]
        _run_ffmpeg(cmd1)
        print("[render] Pass 1 complete")

    # --- Pass 2: Burn ASS subtitles ---
    if not has_subs:
        shutil.copy2(pass1_out, output_path)
        return output_path

    vf_parts = []
    if caption_ass:
        vf_parts.append(f"ass='{_ass_filter_path(caption_ass)}'")
    if hook_ass:
        vf_parts.append(f"ass='{_ass_filter_path(hook_ass)}'")

    print("[render] Pass 2: burning subtitles...")
    cmd2 = [
        "ffmpeg", "-i", pass1_out,
        "-vf", ",".join(vf_parts),
        "-c:v", "libx264", "-crf", str(crf),
        "-c:a", "copy", "-y", output_path,
    ]
    _run_ffmpeg(cmd2)
    print("[render] Pass 2 complete")
    return output_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def enhance(input_path: str, output_path: str, config: dict,
            platform: str | None = None) -> str:
    """Full enhancement pipeline: transcribe → analyze → render."""
    tmp_dir = tempfile.mkdtemp(prefix="quickcut_")

    try:
        # 1. Transcribe
        transcript = transcribe(input_path, config)

        # 2. Analyze and fetch B-roll
        broll_clips = []
        if config.get("broll", {}).get("enabled", True):
            moments = analyze_broll(transcript, config)
            source = config.get("broll", {}).get("source", "ai")
            clip_dur = config.get("broll", {}).get("clip_duration", 3.5)

            for moment in moments:
                try:
                    if source == "pexels":
                        clip_path = fetch_pexels(moment["keyword"], config)
                    else:
                        clip_path = fetch_ai_broll(
                            moment["keyword"], moment.get("reason", ""),
                            clip_dur, config)
                    broll_clips.append({**moment, "clip_path": clip_path})
                except (RuntimeError, httpx.HTTPError, subprocess.CalledProcessError,
                        json.JSONDecodeError, OSError) as e:
                    print(f"[broll] Failed to fetch '{moment['keyword']}': {e}")

        # 3. Plan dynamic edits
        edit_points = []
        if config.get("dynamic_edits", {}).get("enabled", True):
            edit_points = plan_edits(transcript["duration"], broll_clips, config)
            print(f"[edits] Planned {len(edit_points)} edit points")

        # 4. Build montage
        montage_path, montage_dur = "", 0.0
        if config.get("montage", {}).get("enabled", True):
            montage_path, montage_dur = build_montage(config, tmp_dir)

        # 5. Generate ASS files
        caption_ass = str(Path(tmp_dir) / "captions.ass")
        build_caption_ass(transcript, caption_ass, config, platform)

        hook_ass = ""
        if config.get("hook_text", {}).get("enabled", True):
            hook = extract_hook(transcript)
            if hook:
                hook_ass = str(Path(tmp_dir) / "hook.ass")
                build_hook_ass(hook, hook_ass, config, platform)

        # 6. Render
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        render_enhanced(
            input_path, broll_clips, edit_points,
            montage_path, montage_dur,
            caption_ass, hook_ass,
            output_path, config, tmp_dir,
        )

        print(f"\n[done] Output: {output_path}")
        return output_path

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuickCut \u2014 2-pass clip enhancer")
    parser.add_argument("--input", "-i", required=True, help="Path to input clip")
    parser.add_argument("--output", "-o", default=None, help="Output path")
    parser.add_argument("--config", "-c", default="../video-forge-lite/config.yaml")
    parser.add_argument("--broll-source", choices=["ai", "pexels"], default=None)
    parser.add_argument("--platform", "-p", default=None,
                        choices=["tiktok", "instagram", "youtube_shorts", "x_twitter", "facebook"])
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    config = load_config(args.config)

    if args.broll_source:
        config.setdefault("broll", {})["source"] = args.broll_source

    output = args.output or f"output/enhanced/{Path(args.input).stem}_quickcut.mp4"
    enhance(args.input, output, config, args.platform)
