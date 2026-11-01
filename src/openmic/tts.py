"""SiliconFlow CosyVoice TTS adapter."""

import io
import os
import re
import struct
import time
import wave
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv


SYSTEM_VOICES: Dict[str, str] = {
    "Alex｜沉稳男声": "alex",
    "Benjamin｜低沉男声": "benjamin",
    "Charles｜磁性男声": "charles",
    "David｜欢快男声": "david",
    "Anna｜沉稳女声": "anna",
    "Bella｜激情女声": "bella",
    "Claire｜温柔女声": "claire",
    "Diana｜欢快女声": "diana",
}


LEADING_BOILERPLATE = (
    "表演指导已到位",
    "以下是经过表演标记",
    "以下是最终脚本",
    "可直接用于 tts",
    "所有标记均已内嵌",
)

TRAILING_BOILERPLATE = (
    "如需调整",
    "请直接告诉我",
    "我来微调",
    "以上为表演",
    "以上是表演",
)


@dataclass(frozen=True)
class TTSConfig:
    api_key: str
    base_url: str
    model: str = "FunAudioLLM/CosyVoice2-0.5B"
    voice: str = "FunAudioLLM/CosyVoice2-0.5B:alex"
    response_format: str = "wav"
    sample_rate: int = 16000
    timeout_seconds: int = 240

    @classmethod
    def from_env(cls) -> "TTSConfig":
        load_dotenv()
        api_key = (
            os.getenv("TTS_API_KEY")
            or os.getenv("OPENMIC_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("缺少 TTS_API_KEY 或 LLM_API_KEY")

        base_url = (
            os.getenv("TTS_BASE_URL")
            or os.getenv("OPENMIC_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.siliconflow.cn/v1"
        )
        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            model=os.getenv("TTS_MODEL", "FunAudioLLM/CosyVoice2-0.5B"),
            voice=os.getenv(
                "TTS_VOICE", "FunAudioLLM/CosyVoice2-0.5B:alex"
            ),
            response_format=os.getenv("TTS_FORMAT", "wav"),
            sample_rate=int(os.getenv("TTS_SAMPLE_RATE", "16000")),
        )


@dataclass(frozen=True)
class TTSResult:
    audio: bytes
    media_type: str
    model: str
    voice: str
    sample_rate: int
    channels: Optional[int] = None
    duration_seconds: Optional[float] = None
    generation_seconds: Optional[float] = None


class TTSRequestError(RuntimeError):
    pass


def inspect_wav(audio: bytes) -> Tuple[int, int, int, float]:
    """Return ``(sample_rate, channels, sample_width, duration_seconds)``.

    The provider response is treated as untrusted bytes.  Parsing the header
    here gives the UI and later evaluation code one reliable source of audio
    metadata, and lets us fail early when a provider returns a non-WAV payload
    or an unexpected sampling rate.
    """
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
    except (EOFError, wave.Error, ValueError) as exc:
        raise TTSRequestError("TTS 返回的内容不是有效的 WAV 音频") from exc

    if sample_rate <= 0 or channels <= 0 or sample_width <= 0:
        raise TTSRequestError("TTS 返回的 WAV 音频头信息无效")
    return (
        sample_rate,
        channels,
        sample_width,
        frame_count / sample_rate,
    )


def extract_spoken_script(text: str) -> str:
    """Extract only the speakable script from a PerformanceCoach response."""
    boundary_match = re.search(
        r"<SCRIPT>\s*(.*?)\s*</SCRIPT>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if boundary_match:
        return boundary_match.group(1).strip()

    working = text.strip()
    first_marker = re.search(
        r"(?m)^(?:>\s*)?(?=\[(?:PAUSE|SETUP|EMOTION|PUNCHLINE|CALLBACK|EMPHASIS)[=\]])",
        working,
    )
    if first_marker and any(
        marker in working[: first_marker.start()].lower()
        for marker in LEADING_BOILERPLATE
    ):
        working = working[first_marker.start() :]

    trailing_positions = []
    for marker in TRAILING_BOILERPLATE:
        match = re.search(rf"(?m)^\s*{re.escape(marker)}", working, flags=re.IGNORECASE)
        if match:
            trailing_positions.append(match.start())
    if trailing_positions:
        working = working[: min(trailing_positions)].rstrip()

    paragraphs = [
        part.strip() for part in re.split(r"\n\s*\n", working) if part.strip()
    ]
    while paragraphs and any(
        marker in paragraphs[0].lower() for marker in LEADING_BOILERPLATE
    ):
        paragraphs.pop(0)
    while paragraphs and any(
        marker in paragraphs[-1].lower() for marker in TRAILING_BOILERPLATE
    ):
        paragraphs.pop()

    # Older responses sometimes put the preamble and first marker in one paragraph.
    if paragraphs:
        marker_match = re.search(
            r"(?m)^(?:>\s*)?(?=\[(?:PAUSE|SETUP|EMOTION|PUNCHLINE|CALLBACK|EMPHASIS)[=\]])",
            paragraphs[0],
        )
        if marker_match:
            paragraphs[0] = paragraphs[0][marker_match.start() :].strip()

    return "\n\n".join(paragraphs).strip()


def prepare_tts_text(text: str) -> str:
    """Convert internal performance markers into CosyVoice-friendly prose."""
    spoken_script = extract_spoken_script(text)
    cleaned = re.sub(r"\[PAUSE=\d+(?:\.\d+)?\]", "……", spoken_script)
    cleaned = re.sub(r"\[(?:SETUP|PUNCHLINE|CALLBACK|EMPHASIS)\]", "", cleaned)
    cleaned = re.sub(r"\[EMOTION=[^\]]+\]", "", cleaned)
    cleaned = cleaned.replace("**", "").replace("`", "")
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^>\s?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    prompt = "请用幽默、自嘲、富有舞台感的中文朗读，包袱前适当停顿，重点词加强语气。"
    return f"{prompt}<|endofprompt|>{cleaned}"


def normalize_wav_header(audio: bytes) -> bytes:
    """Replace streaming-size placeholders with real RIFF and data sizes."""
    if len(audio) < 12 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        return audio

    normalized = bytearray(audio)
    struct.pack_into("<I", normalized, 4, len(normalized) - 8)
    offset = 12
    while offset + 8 <= len(normalized):
        chunk_id = bytes(normalized[offset : offset + 4])
        chunk_size = struct.unpack_from("<I", normalized, offset + 4)[0]
        if chunk_id == b"data":
            struct.pack_into("<I", normalized, offset + 4, len(normalized) - offset - 8)
            break
        next_offset = offset + 8 + chunk_size + (chunk_size % 2)
        if next_offset <= offset or next_offset > len(normalized):
            break
        offset = next_offset
    return bytes(normalized)


class SiliconFlowTTS:
    def __init__(self, config: TTSConfig, session: Optional[Any] = None):
        self.config = config
        self.session = session or requests.Session()

    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> TTSResult:
        if not text.strip():
            raise ValueError("TTS 文本不能为空")
        if not 0.25 <= speed <= 4.0:
            raise ValueError("speed 必须在 0.25 到 4.0 之间")

        selected_voice = voice or self.config.voice
        prepared_text = prepare_tts_text(text)
        spoken_text = prepared_text.split("<|endofprompt|>", 1)[-1].strip()
        if not spoken_text:
            raise ValueError("TTS 文本清理后为空")
        payload = {
            "model": self.config.model,
            "voice": selected_voice,
            "input": prepared_text,
            "response_format": self.config.response_format,
            "sample_rate": self.config.sample_rate,
            "speed": speed,
            "gain": 0,
        }
        started = time.perf_counter()
        response = self.session.post(
            f"{self.config.base_url}/audio/speech",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        generation_seconds = time.perf_counter() - started
        if response.status_code != 200:
            body = response.text[:500].replace(self.config.api_key, "[REDACTED]")
            raise TTSRequestError(
                f"TTS 请求失败（HTTP {response.status_code}）：{body}"
            )

        media_type = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "opus": "audio/ogg",
            "pcm": "audio/pcm",
        }.get(self.config.response_format, "application/octet-stream")
        audio = response.content
        channels: Optional[int] = None
        duration_seconds: Optional[float] = None
        if self.config.response_format == "wav":
            audio = normalize_wav_header(audio)
            actual_rate, channels, _sample_width, duration_seconds = inspect_wav(audio)
            if actual_rate != self.config.sample_rate:
                raise TTSRequestError(
                    "TTS 返回采样率不符合配置："
                    f"期望 {self.config.sample_rate} Hz，实际 {actual_rate} Hz"
                )
        return TTSResult(
            audio=audio,
            media_type=media_type,
            model=self.config.model,
            voice=selected_voice,
            sample_rate=self.config.sample_rate,
            channels=channels,
            duration_seconds=duration_seconds,
            generation_seconds=generation_seconds,
        )
