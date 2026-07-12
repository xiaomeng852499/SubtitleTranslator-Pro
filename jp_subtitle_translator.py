#!/usr/bin/env python3
"""Offline Japanese subtitle translator powered by local Ollama models."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:7b"
FALLBACK_MODEL = "hunyuan-mt"
MODEL_OPTIONS = (
    "qwen2.5:7b",
    "qwen2.5",
    "qwen2.5:latest",
    "qwen2.5vl:7b",
    "hunyuan-mt",
    "qwen25coder14b",
)
DEFAULT_NUM_GPU = 999
DEFAULT_TIMEOUT = 600
DEFAULT_RETRIES = 3
DEFAULT_BATCH_SIZE = 30
DEFAULT_NUM_CTX = 4096
CACHE_FILE_NAME = "translation_cache.json"
CACHE_VERSION = 1
LOCAL_CUDA_BIN = Path(__file__).with_name("cuda_runtime") / "bin"


@dataclass
class SubtitleItem:
    index: int
    text: str
    kind: str
    start: int | None = None
    end: int | None = None
    line_index: int | None = None
    ass_fields: list[str] | None = None


class SubtitleParseError(RuntimeError):
    pass


def read_text_file(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp932", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig", newline="")


def prepare_local_cuda_runtime() -> None:
    if not LOCAL_CUDA_BIN.exists():
        return
    cuda_bin = str(LOCAL_CUDA_BIN)
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(cuda_bin)
        except OSError:
            pass
    path_value = os.environ.get("PATH", "")
    if cuda_bin.lower() not in path_value.lower():
        os.environ["PATH"] = cuda_bin + os.pathsep + path_value


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes}分{secs}秒"
    if minutes:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


def strip_subtitle_markup(text: str) -> str:
    text = re.sub(r"\{[^}]*\}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.replace("\\N", "\n").strip()


def should_translate(text: str) -> bool:
    compact = strip_subtitle_markup(text)
    if not compact:
        return False
    if re.fullmatch(r"[\s\d:.,\-–—>]+", compact):
        return False
    return True


def parse_srt_or_vtt(content: str) -> tuple[list[str], list[SubtitleItem], str]:
    lines = content.splitlines()
    items: list[SubtitleItem] = []
    i = 0
    if lines and lines[0].lstrip("\ufeff").strip().upper() == "WEBVTT":
        i = 1

    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        block_start = i
        if "-->" not in lines[i] and i + 1 < len(lines) and "-->" in lines[i + 1]:
            i += 1

        if i >= len(lines) or "-->" not in lines[i]:
            i += 1
            continue

        text_start = i + 1
        j = text_start
        while j < len(lines) and lines[j].strip():
            j += 1

        original = "\n".join(lines[k] for k in range(text_start, j))
        if should_translate(original):
            items.append(
                SubtitleItem(
                    index=len(items) + 1,
                    text=original,
                    kind="line",
                    start=text_start,
                    end=j,
                )
            )

        i = max(j + 1, block_start + 1)

    return lines, items, "line"


def split_ass_dialogue_payload(payload: str) -> list[str]:
    fields: list[str] = []
    current: list[str] = []
    comma_count = 0
    for char in payload:
        if char == "," and comma_count < 9:
            fields.append("".join(current))
            current = []
            comma_count += 1
        else:
            current.append(char)
    fields.append("".join(current))
    return fields


def parse_ass_or_ssa(content: str) -> tuple[list[str], list[SubtitleItem], str]:
    lines = content.splitlines()
    items: list[SubtitleItem] = []

    for line_index, line in enumerate(lines):
        if not line.startswith("Dialogue:"):
            continue

        payload = line[len("Dialogue:") :].lstrip()
        fields = split_ass_dialogue_payload(payload)
        if len(fields) < 10:
            continue

        text = fields[9].replace("\\N", "\n")
        if not should_translate(text):
            continue

        items.append(
            SubtitleItem(
                index=len(items) + 1,
                text=text,
                kind="ass",
                line_index=line_index,
                ass_fields=fields,
            )
        )

    return lines, items, "ass"


def parse_subtitle(path: Path) -> tuple[list[str], list[SubtitleItem], str]:
    content = read_text_file(path)
    suffix = path.suffix.lower()
    if suffix in {".ass", ".ssa"}:
        return parse_ass_or_ssa(content)
    if suffix in {".srt", ".vtt"}:
        return parse_srt_or_vtt(content)
    raise SubtitleParseError(f"不支持的字幕格式：{suffix}")


def join_subtitle(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


def format_srt_timestamp(seconds: float) -> str:
    millis = int(round(max(0.0, seconds) * 1000))
    hours, remainder = divmod(millis, 3600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt_segments(output_path: Path, segments: list[tuple[float, float, str]]) -> None:
    lines: list[str] = []
    for index, (start, end, text) in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}")
        lines.append(text.strip())
        lines.append("")
    write_text_file(output_path, "\n".join(lines))


def resolve_local_tool(name: str) -> str:
    candidates = [
        Path(__file__).with_name(name),
        Path(__file__).with_name("tools") / name,
        Path(__file__).with_name("ffmpeg") / "bin" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return name


def parse_ffmpeg_out_time(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() == "N/A":
        return None
    if value.isdigit():
        # FFmpeg progress commonly reports microseconds here.
        return int(value) / 1_000_000
    match = re.match(r"(?P<h>\d+):(?P<m>\d+):(?P<s>\d+(?:\.\d+)?)", value)
    if not match:
        return None
    return (
        int(match.group("h")) * 3600
        + int(match.group("m")) * 60
        + float(match.group("s"))
    )


def run_ffmpeg_extract_attempt(
    command: list[str],
    wav_path: Path,
    duration: float | None,
    progress: Callable[[str], None] | None,
) -> tuple[bool, int, str]:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    output_lines: list[str] = []
    last_percent = -1
    last_report_at = 0.0

    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        output_lines.append(line)
        if len(output_lines) > 200:
            del output_lines[:100]

        if line.startswith("out_time_ms=") or line.startswith("out_time_us="):
            seconds = parse_ffmpeg_out_time(line.split("=", 1)[1])
        elif line.startswith("out_time="):
            seconds = parse_ffmpeg_out_time(line.split("=", 1)[1])
        else:
            seconds = None

        if progress and duration and seconds is not None:
            percent = max(0, min(100, int(seconds / duration * 100)))
            now = time.perf_counter()
            if percent >= last_percent + 2 or now - last_report_at >= 10:
                last_percent = percent
                last_report_at = now
                progress(f"ffmpeg 提取音频进度：约 {percent}%")

    returncode = process.wait()
    ok = returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 44
    return ok, returncode, "\n".join(output_lines[-80:])


def extract_audio_with_ffmpeg(
    input_path: Path,
    wav_path: Path,
    audio_enhance: bool = False,
    duration: float | None = None,
    progress: Callable[[str], None] | None = None,
) -> None:
    ffmpeg = resolve_local_tool("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    base_output = [
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
    ]
    if audio_enhance:
        base_output.extend(["-af", "highpass=f=80,lowpass=f=7800,loudnorm=I=-16:TP=-1.5:LRA=11"])

    attempts = [
        (
            "标准模式",
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-y",
                "-progress",
                "pipe:1",
                "-loglevel",
                "warning",
                "-i",
                str(input_path),
                *base_output,
                str(wav_path),
            ],
        ),
        (
            "容错模式",
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-y",
                "-progress",
                "pipe:1",
                "-loglevel",
                "warning",
                "-fflags",
                "+genpts+discardcorrupt",
                "-err_detect",
                "ignore_err",
                "-analyzeduration",
                "100M",
                "-probesize",
                "100M",
                "-i",
                str(input_path),
                *base_output,
                str(wav_path),
            ],
        ),
        (
            "兜底模式",
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-y",
                "-progress",
                "pipe:1",
                "-loglevel",
                "warning",
                "-i",
                str(input_path),
                "-vn",
                "-sn",
                "-dn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(wav_path),
            ],
        ),
    ]

    last_error = ""
    last_code = 0
    for label, command in attempts:
        if wav_path.exists():
            wav_path.unlink()
        if progress:
            progress(f"ffmpeg 正在提取音频：{label}")
        try:
            ok, last_code, last_error = run_ffmpeg_extract_attempt(command, wav_path, duration, progress)
        except FileNotFoundError as exc:
            raise RuntimeError("未找到 ffmpeg。请把 ffmpeg.exe 放到本程序目录、tools 目录，或加入系统 PATH。") from exc
        if ok:
            if progress:
                size_mb = wav_path.stat().st_size / 1024 / 1024
                progress(f"音频提取完成：{size_mb:.1f} MB")
            return
        if progress:
            progress(f"{label}失败，退出码：{last_code}，准备尝试下一种方式。")

    if "received signal 15" in last_error.lower():
        raise RuntimeError(
            "ffmpeg 提取音频被系统或程序终止（signal 15）。已自动重试但仍失败。"
            "请关闭音频增强、确认电脑不会睡眠，并检查杀毒软件是否拦截 ffmpeg。"
        )
    detail = last_error.strip().splitlines()[-1:] or [f"ffmpeg exit code {last_code}"]
    raise RuntimeError(f"ffmpeg 提取音频失败：{detail[0]}")


def probe_media_duration(input_path: Path) -> float | None:
    command = [
        resolve_local_tool("ffprobe.exe" if os.name == "nt" else "ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def is_cuda_library_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "cublas",
            "cudnn",
            "cuda",
            "cannot be loaded",
            "not found or cannot be loaded",
        )
    )


def transcribe_wav_to_segments(
    wav_path: Path,
    whisper_model: str,
    language: str,
    device: str,
    compute_type: str,
    recognition_mode: str,
    duration: float | None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[tuple[float, float, str]], str]:
    from faster_whisper import WhisperModel

    if progress:
        progress(f"正在加载 faster-whisper 模型：{whisper_model} / {device} / {compute_type}")
        if whisper_model.lower() == "large-v3":
            progress("提示：large-v3 加载和识别都比较慢，3070 长视频建议优先用 medium 或 small。")
    model = WhisperModel(whisper_model, device=device, compute_type=compute_type)

    if progress:
        progress("正在识别语音...")
        progress(f"识别模式：{recognition_mode}")
    mode = recognition_mode.strip().lower()
    if mode in {"高准确率", "accurate", "high", "high_accuracy"}:
        transcribe_options = {
            "beam_size": 8,
            "best_of": 5,
            "temperature": 0,
            "vad_filter": True,
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.5,
        }
    elif mode in {"快速", "fast"}:
        transcribe_options = {
            "beam_size": 1,
            "best_of": 1,
            "temperature": 0,
            "vad_filter": True,
            "condition_on_previous_text": False,
        }
    else:
        transcribe_options = {
            "beam_size": 5,
            "best_of": 3,
            "temperature": 0,
            "vad_filter": True,
            "condition_on_previous_text": True,
        }
    segments_iter, info = model.transcribe(
        str(wav_path),
        language=language or None,
        **transcribe_options,
    )

    segments: list[tuple[float, float, str]] = []
    last_reported_percent = -1
    for segment in segments_iter:
        text = segment.text.strip()
        if not text:
            continue
        segments.append((float(segment.start), float(segment.end), text))
        if progress:
            if duration:
                percent = min(100, int((float(segment.end) / duration) * 100))
                if percent >= last_reported_percent + 5 or len(segments) % 20 == 0:
                    last_reported_percent = percent
                    progress(f"已识别 {len(segments)} 条字幕，进度约 {percent}%")
            elif len(segments) % 20 == 0:
                progress(f"已识别 {len(segments)} 条字幕...")

    return segments, str(getattr(info, "language", language))


def generate_subtitle_from_media(
    input_path: Path,
    output_path: Path,
    whisper_model: str = "medium",
    language: str = "ja",
    device: str = "cuda",
    compute_type: str = "float16",
    recognition_mode: str = "标准",
    audio_enhance: bool = False,
    progress: Callable[[str], None] | None = None,
) -> int:
    prepare_local_cuda_runtime()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("未找到 faster-whisper。请先安装 faster-whisper。") from exc

    if progress:
        progress("正在提取音频...")
    duration = probe_media_duration(input_path)
    if progress and duration:
        progress(f"媒体时长：{format_duration(duration)}")

    with tempfile.TemporaryDirectory(prefix="subtitle_audio_") as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        if progress and audio_enhance:
            progress("音频增强：开启")
        extract_audio_with_ffmpeg(
            input_path,
            wav_path,
            audio_enhance=audio_enhance,
            duration=duration,
            progress=progress,
        )

        try:
            segments, detected_language = transcribe_wav_to_segments(
                wav_path,
                whisper_model,
                language,
                device,
                compute_type,
                recognition_mode,
                duration,
                progress,
            )
        except Exception as exc:
            if device.lower() == "cuda" and is_cuda_library_error(exc):
                if progress:
                    progress(f"CUDA 加载失败：{exc}")
                    progress("自动改用 CPU / int8 重试。速度会慢一些，但可以先生成字幕。")
                segments, detected_language = transcribe_wav_to_segments(
                    wav_path,
                    whisper_model,
                    language,
                    "cpu",
                    "int8",
                    recognition_mode,
                    duration,
                    progress,
                )
            else:
                raise

    write_srt_segments(output_path, segments)
    if progress:
        progress(f"字幕生成完成：{output_path}，共 {len(segments)} 条，语言：{detected_language}")
    return len(segments)


def apply_translations(base_lines: list[str], items: list[SubtitleItem], translations: dict[str, str]) -> list[str]:
    lines = list(base_lines)
    item_map = {str(item.index): item for item in items}
    for key in sorted(translations, key=lambda value: int(value), reverse=True):
        item = item_map.get(key)
        if not item:
            continue
        translated = translations[key].strip()
        if not translated:
            continue

        if item.kind == "line":
            replacement = [line.strip() for line in translated.splitlines() if line.strip()]
            if not replacement:
                replacement = [""]
            assert item.start is not None and item.end is not None
            lines[item.start : item.end] = replacement
        elif item.kind == "ass":
            assert item.line_index is not None and item.ass_fields is not None
            fields = list(item.ass_fields)
            fields[9] = translated.replace("\n", "\\N")
            lines[item.line_index] = "Dialogue: " + ",".join(fields)

    return lines


def progress_path_for(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".progress.json")


def default_cache_path() -> Path:
    return Path(__file__).with_name(CACHE_FILE_NAME)


def load_progress(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(read_text_file(path))
    except json.JSONDecodeError:
        return {}
    translations = payload.get("translations", {})
    if isinstance(translations, dict):
        return {str(key): str(value) for key, value in translations.items()}
    return {}


def save_progress(path: Path, input_path: Path, output_path: Path, model: str, translations: dict[str, str]) -> None:
    payload = {
        "input": str(input_path),
        "output": str(output_path),
        "model": model,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "translations": translations,
    }
    write_text_file(path, json.dumps(payload, ensure_ascii=False, indent=2))


def normalize_cache_source(text: str) -> str:
    return one_line_text(text)


def cache_key(model: str, bilingual: bool, source_text: str) -> str:
    mode = "bilingual" if bilingual else "zh"
    return f"{CACHE_VERSION}|{model.strip()}|{mode}|{normalize_cache_source(source_text)}"


def load_translation_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(read_text_file(path))
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("entries"), dict):
        return {str(key): str(value) for key, value in payload["entries"].items()}
    if isinstance(payload, dict):
        return {str(key): str(value) for key, value in payload.items()}
    return {}


def save_translation_cache(path: Path, entries: dict[str, str]) -> None:
    payload = {
        "version": CACHE_VERSION,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entries": entries,
    }
    write_text_file(path, json.dumps(payload, ensure_ascii=False, indent=2))


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def list_models(self) -> list[str]:
        request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return [item.get("name", "") for item in payload.get("models", []) if item.get("name")]

    def generate(
        self,
        model: str,
        prompt: str,
        num_gpu: int | None = DEFAULT_NUM_GPU,
        num_ctx: int = DEFAULT_NUM_CTX,
    ) -> str:
        options: dict[str, int | float] = {
            "temperature": 0,
            "num_ctx": num_ctx,
        }
        if num_gpu is not None:
            options["num_gpu"] = num_gpu

        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": options,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("response", "").strip()


def one_line_text(text: str) -> str:
    return re.sub(r"\s+", " ", strip_subtitle_markup(text)).strip()


def build_prompt(text: str, bilingual: bool) -> str:
    mode = "输出日文原文和简体中文译文。" if bilingual else "只输出简体中文译文。"
    return (
        "日语字幕翻译成简体中文。\n"
        "不要解释，不要扩写。"
        f"{mode}\n\n"
        f"{text}"
    )


def build_batch_prompt(items: list[SubtitleItem], bilingual: bool) -> str:
    mode = "译文可包含日文原文，但必须放在同一编号后。" if bilingual else "只写中文译文。"
    lines = "\n".join(f"{item.index}. {one_line_text(item.text)}" for item in items)
    return (
        "把下列日语字幕译成简体中文。\n"
        "只返回：编号. 译文。不要解释，不要总结，不要加标题。\n"
        f"{mode}\n"
        f"{lines}"
    )


def clean_model_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^翻译[:：]\s*", "", text)
    text = re.sub(r"^译文[:：]\s*", "", text)
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_numbered_translations(text: str, expected_indexes: set[int]) -> dict[str, str]:
    text = clean_model_output(text)
    parsed: dict[str, str] = {}
    current_key: str | None = None
    pattern = re.compile(r"^\s*[\[\(【]?\s*(\d+)\s*[\]\)】]?\s*[\.。:：、\-]\s*(.+?)\s*$")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            index = int(match.group(1))
            if index in expected_indexes:
                current_key = str(index)
                parsed[current_key] = match.group(2).strip()
            else:
                current_key = None
        elif current_key:
            parsed[current_key] = f"{parsed[current_key]}\n{line}".strip()

    return {key: value for key, value in parsed.items() if value}


def is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, socket.timeout):
        return True
    if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError):
        return True
    return "timed out" in str(exc).lower()


def translate_one_with_retry(
    item: SubtitleItem,
    client: OllamaClient,
    model: str,
    bilingual: bool,
    num_gpu: int | None,
    retries: int,
    progress: Callable[[str], None] | None,
) -> str:
    prompt = build_prompt(strip_subtitle_markup(item.text), bilingual)
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            return clean_model_output(client.generate(model, prompt, num_gpu=num_gpu))
        except Exception as exc:
            last_error = exc
            if progress:
                reason = "超时" if is_timeout_error(exc) else str(exc)
                progress(f"[{item.index}] 第 {attempt}/{retries} 次失败：{reason}")
            if attempt < retries:
                time.sleep(min(2 * attempt, 8))
    assert last_error is not None
    raise RuntimeError(f"第 {item.index} 条字幕翻译失败，已保存进度。最后错误：{last_error}")


def translate_batch_with_retry(
    batch: list[SubtitleItem],
    client: OllamaClient,
    model: str,
    bilingual: bool,
    num_gpu: int | None,
    retries: int,
    progress: Callable[[str], None] | None,
) -> dict[str, str]:
    expected = {item.index for item in batch}
    prompt = build_batch_prompt(batch, bilingual)
    last_error: BaseException | None = None

    for attempt in range(1, retries + 1):
        try:
            output = client.generate(model, prompt, num_gpu=num_gpu, num_ctx=DEFAULT_NUM_CTX)
            parsed = parse_numbered_translations(output, expected)
            missing = sorted(expected - {int(key) for key in parsed})
            if missing:
                raise ValueError(f"缺少编号：{', '.join(str(item) for item in missing[:8])}")
            return parsed
        except Exception as exc:
            last_error = exc
            if progress:
                reason = "超时" if is_timeout_error(exc) else str(exc)
                first = batch[0].index
                last = batch[-1].index
                progress(f"[{first}-{last}] 第 {attempt}/{retries} 次失败：{reason}")
            if attempt < retries:
                time.sleep(min(2 * attempt, 8))

    assert last_error is not None
    first = batch[0].index
    last = batch[-1].index
    raise RuntimeError(f"第 {first}-{last} 条字幕批量翻译失败，已保存进度。最后错误：{last_error}")


def save_translated_items(
    translated_batch: dict[str, str],
    batch: list[SubtitleItem],
    translations: dict[str, str],
    cache_entries: dict[str, str],
    cache_path: Path,
    input_path: Path,
    output_path: Path,
    model: str,
    bilingual: bool,
    use_cache: bool,
    base_lines: list[str],
    all_items: list[SubtitleItem],
    progress_path: Path,
) -> None:
    translations.update(translated_batch)
    if use_cache:
        for item in batch:
            translated = translated_batch.get(str(item.index), "").strip()
            if translated:
                cache_entries[cache_key(model, bilingual, item.text)] = translated
        save_translation_cache(cache_path, cache_entries)
    save_progress(progress_path, input_path, output_path, model, translations)
    current_lines = apply_translations(base_lines, all_items, translations)
    write_text_file(output_path, join_subtitle(current_lines))


def translate_batch_auto_split(
    batch: list[SubtitleItem],
    client: OllamaClient,
    model: str,
    bilingual: bool,
    num_gpu: int | None,
    retries: int,
    translations: dict[str, str],
    cache_entries: dict[str, str],
    cache_path: Path,
    input_path: Path,
    output_path: Path,
    use_cache: bool,
    base_lines: list[str],
    all_items: list[SubtitleItem],
    progress_path: Path,
    progress: Callable[[str], None] | None,
) -> None:
    if not batch:
        return

    try:
        translated_batch = translate_batch_with_retry(batch, client, model, bilingual, num_gpu, retries, progress)
        save_translated_items(
            translated_batch,
            batch,
            translations,
            cache_entries,
            cache_path,
            input_path,
            output_path,
            model,
            bilingual,
            use_cache,
            base_lines,
            all_items,
            progress_path,
        )
        return
    except Exception as exc:
        if len(batch) == 1:
            item = batch[0]
            if progress:
                progress(f"[{item.index}] 批量失败，改用单条翻译：{exc}")
            translated = translate_one_with_retry(item, client, model, bilingual, num_gpu, retries, progress)
            save_translated_items(
                {str(item.index): translated},
                [item],
                translations,
                cache_entries,
                cache_path,
                input_path,
                output_path,
                model,
                bilingual,
                use_cache,
                base_lines,
                all_items,
                progress_path,
            )
            return

        mid = len(batch) // 2
        first = batch[0].index
        last = batch[-1].index
        if progress:
            progress(f"[{first}-{last}] 批量失败，自动拆成 {len(batch[:mid])} + {len(batch[mid:])} 条重试")
        translate_batch_auto_split(
            batch[:mid],
            client,
            model,
            bilingual,
            num_gpu,
            retries,
            translations,
            cache_entries,
            cache_path,
            input_path,
            output_path,
            use_cache,
            base_lines,
            all_items,
            progress_path,
            progress,
        )
        translate_batch_auto_split(
            batch[mid:],
            client,
            model,
            bilingual,
            num_gpu,
            retries,
            translations,
            cache_entries,
            cache_path,
            input_path,
            output_path,
            use_cache,
            base_lines,
            all_items,
            progress_path,
            progress,
        )


def translate_file(
    input_path: Path,
    output_path: Path,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    bilingual: bool = False,
    num_gpu: int | None = DEFAULT_NUM_GPU,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume: bool = True,
    use_cache: bool = True,
    cache_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> int:
    base_lines, items, _mode = parse_subtitle(input_path)
    if progress:
        progress(f"读取完成：找到 {len(items)} 条可翻译字幕")

    progress_path = progress_path_for(output_path)
    translations = load_progress(progress_path) if resume else {}
    if translations and progress:
        progress(f"已载入断点进度：{len(translations)} 条，继续翻译剩余部分")

    if not items:
        write_text_file(output_path, join_subtitle(base_lines))
        return 0

    client = OllamaClient(base_url, timeout=timeout)
    total = len(items)
    batch_size = max(1, min(batch_size, 80))
    pending = [item for item in items if not translations.get(str(item.index), "").strip()]
    cache_path = cache_path or default_cache_path()
    cache_entries = load_translation_cache(cache_path) if use_cache else {}

    cache_hits = 0
    if use_cache and pending:
        still_pending: list[SubtitleItem] = []
        for item in pending:
            key = cache_key(model, bilingual, item.text)
            cached = cache_entries.get(key, "").strip()
            if cached:
                translations[str(item.index)] = cached
                cache_hits += 1
            else:
                still_pending.append(item)
        pending = still_pending
        if cache_hits:
            save_progress(progress_path, input_path, output_path, model, translations)
            current_lines = apply_translations(base_lines, items, translations)
            write_text_file(output_path, join_subtitle(current_lines))
            if progress:
                progress(f"缓存命中：{cache_hits} 条，已跳过模型翻译")

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        if not batch:
            continue

        if progress:
            first = batch[0].index
            last = batch[-1].index
            progress(f"[{first}-{last}/{total}] 批量翻译 {len(batch)} 条")

        translate_batch_auto_split(
            batch,
            client,
            model,
            bilingual,
            num_gpu,
            retries,
            translations,
            cache_entries,
            cache_path,
            input_path,
            output_path,
            use_cache,
            base_lines,
            items,
            progress_path,
            progress,
        )

    final_lines = apply_translations(base_lines, items, translations)
    write_text_file(output_path, join_subtitle(final_lines))
    if progress_path.exists():
        progress_path.unlink()
    if progress:
        progress(f"完成：{output_path}")
    return len(translations)


class TranslatorApp:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.ttk = ttk
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.generate_button = None

        self.root = tk.Tk()
        self.root.title("日语字幕离线翻译")
        self.root.geometry("780x570")
        self.root.minsize(700, 500)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.url_var = tk.StringVar(value=DEFAULT_OLLAMA_URL)
        self.num_gpu_var = tk.StringVar(value=str(DEFAULT_NUM_GPU))
        self.timeout_var = tk.StringVar(value=str(DEFAULT_TIMEOUT))
        self.retries_var = tk.StringVar(value=str(DEFAULT_RETRIES))
        self.batch_size_var = tk.StringVar(value=str(DEFAULT_BATCH_SIZE))
        self.bilingual_var = tk.BooleanVar(value=False)
        self.resume_var = tk.BooleanVar(value=True)
        self.cache_var = tk.BooleanVar(value=True)

        self.build_ui()
        self.root.after(120, self.drain_events)

    def build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk

        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(11, weight=1)

        ttk.Label(frame, text="输入字幕").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="选择", command=self.choose_input).grid(row=0, column=2)

        ttk.Label(frame, text="输出文件").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="另存为", command=self.choose_output).grid(row=1, column=2)

        ttk.Label(frame, text="Ollama 地址").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.url_var).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="检测模型", command=self.check_models).grid(row=2, column=2)

        ttk.Label(frame, text="模型名").grid(row=3, column=0, sticky="w", pady=6)
        model_box = ttk.Combobox(frame, textvariable=self.model_var, values=MODEL_OPTIONS)
        model_box.grid(row=3, column=1, sticky="ew", padx=8)

        ttk.Label(frame, text="GPU 层数").grid(row=4, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.num_gpu_var).grid(row=4, column=1, sticky="ew", padx=8)

        ttk.Label(frame, text="单条超时秒数").grid(row=5, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.timeout_var).grid(row=5, column=1, sticky="ew", padx=8)

        ttk.Label(frame, text="失败重试次数").grid(row=6, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.retries_var).grid(row=6, column=1, sticky="ew", padx=8)

        ttk.Label(frame, text="每批字幕数").grid(row=7, column=0, sticky="w", pady=6)
        ttk.Entry(frame, textvariable=self.batch_size_var).grid(row=7, column=1, sticky="ew", padx=8)

        options = ttk.Frame(frame)
        options.grid(row=8, column=1, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(options, text="双语字幕（日文+中文）", variable=self.bilingual_var).pack(side="left")
        ttk.Checkbutton(options, text="断点续跑", variable=self.resume_var).pack(side="left", padx=(18, 0))
        ttk.Checkbutton(options, text="使用翻译缓存", variable=self.cache_var).pack(side="left", padx=(18, 0))

        self.start_button = ttk.Button(frame, text="开始翻译", command=self.start_translate)
        self.start_button.grid(row=9, column=1, sticky="ew", padx=8, pady=12)
        ttk.Button(frame, text="生成字幕", command=self.open_generate_window).grid(row=9, column=2, sticky="ew", pady=12)

        self.status_var = tk.StringVar(value="准备就绪。支持 .srt / .vtt / .ass / .ssa")
        ttk.Label(frame, textvariable=self.status_var).grid(row=10, column=0, columnspan=3, sticky="w", pady=(6, 4))

        self.log = tk.Text(frame, height=12, wrap="word")
        self.log.grid(row=11, column=0, columnspan=3, sticky="nsew")

    def choose_input(self) -> None:
        path = self.filedialog.askopenfilename(
            title="选择日语字幕",
            filetypes=[("Subtitle files", "*.srt *.vtt *.ass *.ssa"), ("All files", "*.*")],
        )
        if not path:
            return
        self.input_var.set(path)
        input_path = Path(path)
        self.output_var.set(str(input_path.with_name(f"{input_path.stem}.zh{input_path.suffix}")))

    def choose_output(self) -> None:
        path = self.filedialog.asksaveasfilename(
            title="保存中文字幕",
            defaultextension=".srt",
            filetypes=[("Subtitle files", "*.srt *.vtt *.ass *.ssa"), ("All files", "*.*")],
        )
        if path:
            self.output_var.set(path)

    def open_generate_window(self) -> None:
        tk = self.tk
        ttk = self.ttk

        window = tk.Toplevel(self.root)
        window.title("从视频/音频生成字幕")
        window.geometry("720x300")
        window.minsize(660, 260)
        window.columnconfigure(1, weight=1)

        media_var = tk.StringVar()
        srt_var = tk.StringVar()
        whisper_model_var = tk.StringVar(value="medium")
        language_var = tk.StringVar(value="ja")
        device_var = tk.StringVar(value="cuda")
        compute_var = tk.StringVar(value="float16")
        recognition_mode_var = tk.StringVar(value="标准")
        audio_enhance_var = tk.BooleanVar(value=False)

        def choose_media() -> None:
            path = self.filedialog.askopenfilename(
                title="选择视频或音频",
                filetypes=[
                    ("Media files", "*.mp4 *.mkv *.avi *.mov *.wmv *.mp3 *.wav *.m4a *.flac *.aac"),
                    ("All files", "*.*"),
                ],
            )
            if not path:
                return
            media_var.set(path)
            media_path = Path(path)
            srt_var.set(str(media_path.with_suffix(".ja.srt")))

        def choose_srt() -> None:
            path = self.filedialog.asksaveasfilename(
                title="保存生成的字幕",
                defaultextension=".srt",
                filetypes=[("SRT subtitle", "*.srt"), ("All files", "*.*")],
            )
            if path:
                srt_var.set(path)

        ttk.Label(window, text="视频/音频").grid(row=0, column=0, sticky="w", padx=12, pady=8)
        ttk.Entry(window, textvariable=media_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(window, text="选择", command=choose_media).grid(row=0, column=2, padx=12)

        ttk.Label(window, text="输出字幕").grid(row=1, column=0, sticky="w", padx=12, pady=8)
        ttk.Entry(window, textvariable=srt_var).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(window, text="另存为", command=choose_srt).grid(row=1, column=2, padx=12)

        ttk.Label(window, text="Whisper 模型").grid(row=2, column=0, sticky="w", padx=12, pady=8)
        ttk.Combobox(
            window,
            textvariable=whisper_model_var,
            values=("medium", "small", "base", "tiny", "large-v3"),
        ).grid(row=2, column=1, sticky="ew", padx=8)

        ttk.Label(window, text="语言").grid(row=3, column=0, sticky="w", padx=12, pady=8)
        ttk.Combobox(window, textvariable=language_var, values=("ja", "zh", "en", "")).grid(
            row=3, column=1, sticky="ew", padx=8
        )

        ttk.Label(window, text="设备").grid(row=4, column=0, sticky="w", padx=12, pady=8)
        device_row = ttk.Frame(window)
        device_row.grid(row=4, column=1, sticky="ew", padx=8)
        ttk.Combobox(device_row, textvariable=device_var, values=("cuda", "cpu"), width=12).pack(side="left")
        ttk.Label(device_row, text="精度").pack(side="left", padx=(18, 6))
        ttk.Combobox(device_row, textvariable=compute_var, values=("float16", "int8_float16", "int8"), width=14).pack(
            side="left"
        )

        ttk.Label(window, text="识别模式").grid(row=5, column=0, sticky="w", padx=12, pady=8)
        mode_row = ttk.Frame(window)
        mode_row.grid(row=5, column=1, sticky="ew", padx=8)
        ttk.Combobox(mode_row, textvariable=recognition_mode_var, values=("标准", "高准确率", "快速"), width=14).pack(
            side="left"
        )
        ttk.Checkbutton(mode_row, text="音频增强", variable=audio_enhance_var).pack(side="left", padx=(18, 0))

        button = ttk.Button(window, text="开始生成")
        button.grid(row=6, column=1, sticky="ew", padx=8, pady=16)
        self.generate_button = button

        def start_generate() -> None:
            input_path = Path(media_var.get().strip())
            output_path = Path(srt_var.get().strip())
            if not input_path.exists():
                self.messagebox.showerror("缺少输入", "请选择存在的视频或音频文件。")
                return
            if not output_path:
                self.messagebox.showerror("缺少输出", "请选择输出字幕文件。")
                return

            button.configure(state="disabled")
            self.log.delete("1.0", "end")
            started_at = time.perf_counter()

            def worker() -> None:
                try:
                    count = generate_subtitle_from_media(
                        input_path=input_path,
                        output_path=output_path,
                        whisper_model=whisper_model_var.get().strip() or "medium",
                        language=language_var.get().strip(),
                        device=device_var.get().strip() or "cuda",
                        compute_type=compute_var.get().strip() or "float16",
                        recognition_mode=recognition_mode_var.get().strip() or "标准",
                        audio_enhance=audio_enhance_var.get(),
                        progress=lambda msg: self.events.put(("log", msg)),
                    )
                    elapsed = format_duration(time.perf_counter() - started_at)
                    self.events.put(("generate_done", f"字幕生成完成，共 {count} 条。本次用时：{elapsed}。"))
                except Exception as exc:
                    elapsed = format_duration(time.perf_counter() - started_at)
                    self.events.put(("generate_error", f"{exc}\n本次已用时：{elapsed}。"))

            threading.Thread(target=worker, daemon=True).start()

        button.configure(command=start_generate)

    def log_line(self, message: str) -> None:
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.status_var.set(message)

    def check_models(self) -> None:
        def worker() -> None:
            try:
                models = OllamaClient(self.url_var.get()).list_models()
                self.events.put(("models", "\n".join(models) if models else "未发现模型"))
            except Exception as exc:
                self.events.put(("error", f"连接 Ollama 失败：{exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def read_positive_int(self, value: str, label: str) -> int:
        try:
            number = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{label} 请输入整数。") from exc
        if number <= 0:
            raise ValueError(f"{label} 必须大于 0。")
        return number

    def start_translate(self) -> None:
        input_path = Path(self.input_var.get().strip())
        output_path = Path(self.output_var.get().strip())
        if not input_path.exists():
            self.messagebox.showerror("缺少输入", "请选择存在的字幕文件。")
            return
        if not output_path:
            self.messagebox.showerror("缺少输出", "请选择输出文件。")
            return

        try:
            num_gpu = int(self.num_gpu_var.get().strip()) if self.num_gpu_var.get().strip() else None
            timeout = self.read_positive_int(self.timeout_var.get(), "单条超时秒数")
            retries = self.read_positive_int(self.retries_var.get(), "失败重试次数")
            batch_size = self.read_positive_int(self.batch_size_var.get(), "每批字幕数")
        except ValueError as exc:
            self.messagebox.showerror("参数无效", str(exc))
            return

        self.start_button.configure(state="disabled")
        self.log.delete("1.0", "end")
        started_at = time.perf_counter()

        def worker() -> None:
            try:
                count = translate_file(
                    input_path=input_path,
                    output_path=output_path,
                    model=self.model_var.get().strip() or DEFAULT_MODEL,
                    base_url=self.url_var.get().strip() or DEFAULT_OLLAMA_URL,
                    bilingual=self.bilingual_var.get(),
                    num_gpu=num_gpu,
                    timeout=timeout,
                    retries=retries,
                    batch_size=batch_size,
                    resume=self.resume_var.get(),
                    use_cache=self.cache_var.get(),
                    progress=lambda msg: self.events.put(("log", msg)),
                )
                elapsed = format_duration(time.perf_counter() - started_at)
                self.events.put(("done", f"翻译完成，共保存 {count} 条字幕。本次用时：{elapsed}。"))
            except Exception as exc:
                elapsed = format_duration(time.perf_counter() - started_at)
                self.events.put(("error", f"{exc}\n本次已用时：{elapsed}。"))

        threading.Thread(target=worker, daemon=True).start()

    def drain_events(self) -> None:
        while True:
            try:
                event, message = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "models":
                self.log_line("本地模型：")
                self.log_line(message)
            elif event == "done":
                self.log_line(message)
                self.start_button.configure(state="normal")
                self.messagebox.showinfo("完成", message)
            elif event == "error":
                self.log_line(message)
                self.start_button.configure(state="normal")
                self.messagebox.showerror("错误", message)
            elif event == "generate_done":
                self.log_line(message)
                if self.generate_button is not None:
                    self.generate_button.configure(state="normal")
                self.messagebox.showinfo("完成", message)
            elif event == "generate_error":
                self.log_line(message)
                if self.generate_button is not None:
                    self.generate_button.configure(state="normal")
                self.messagebox.showerror("错误", message)
            else:
                self.log_line(message)
        self.root.after(120, self.drain_events)

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate Japanese subtitles to Chinese with local Ollama.")
    parser.add_argument("input", nargs="?", help="Input subtitle file: .srt, .vtt, .ass, .ssa")
    parser.add_argument("-o", "--output", help="Output subtitle path")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Ollama model name, default: {DEFAULT_MODEL}")
    parser.add_argument("--url", default=DEFAULT_OLLAMA_URL, help=f"Ollama URL, default: {DEFAULT_OLLAMA_URL}")
    parser.add_argument("--num-gpu", type=int, default=DEFAULT_NUM_GPU, help="GPU layers for Ollama. Use 999 for GPU.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Seconds to wait for each subtitle request.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retry count for each subtitle request.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Subtitles per Ollama request. Try 20-40.")
    parser.add_argument("--no-cache", action="store_true", help="Disable translation cache.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing progress file.")
    parser.add_argument("--bilingual", action="store_true", help="Output Japanese + Chinese bilingual subtitles.")
    parser.add_argument("--transcribe", action="store_true", help="Generate SRT from video/audio with faster-whisper.")
    parser.add_argument("--whisper-model", default="medium", help="faster-whisper model size, default: medium.")
    parser.add_argument("--language", default="ja", help="Speech language code, default: ja.")
    parser.add_argument("--device", default="cuda", help="faster-whisper device: cuda or cpu.")
    parser.add_argument("--compute-type", default="float16", help="faster-whisper compute type.")
    parser.add_argument("--recognition-mode", default="标准", help="Recognition mode: 快速, 标准, 高准确率.")
    parser.add_argument("--no-audio-enhance", action="store_true", help="Disable ffmpeg audio enhancement for transcription.")
    parser.add_argument("--gui", action="store_true", help="Open desktop GUI.")
    args = parser.parse_args()

    if args.gui or not args.input:
        TranslatorApp().run()
        return 0

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}.zh{input_path.suffix}")
    if args.transcribe:
        output_path = Path(args.output) if args.output else input_path.with_suffix(".ja.srt")
        started_at = time.perf_counter()
        count = generate_subtitle_from_media(
            input_path=input_path,
            output_path=output_path,
            whisper_model=args.whisper_model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            recognition_mode=args.recognition_mode,
            audio_enhance=not args.no_audio_enhance,
            progress=print,
        )
        elapsed = format_duration(time.perf_counter() - started_at)
        print(f"Done. Generated {count} subtitle entries -> {output_path}")
        print(f"Elapsed: {elapsed}")
        return 0

    started_at = time.perf_counter()
    count = translate_file(
        input_path=input_path,
        output_path=output_path,
        model=args.model,
        base_url=args.url,
        bilingual=args.bilingual,
        num_gpu=args.num_gpu,
        timeout=args.timeout,
        retries=args.retries,
        batch_size=args.batch_size,
        resume=not args.no_resume,
        use_cache=not args.no_cache,
        progress=print,
    )
    elapsed = format_duration(time.perf_counter() - started_at)
    print(f"Done. Saved {count} subtitle entries -> {output_path}")
    print(f"Elapsed: {elapsed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
