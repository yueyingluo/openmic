"""CosyVoice requests reuse csw's configuration and WAV normalization.

We intentionally bypass prepare_tts_text: its universal performance prompt
would contaminate the neutral baseline and its regex removes our controls.
"""

import hashlib
import io
import struct
import wave
from dataclasses import replace
from typing import Any, Optional

import numpy as np
import requests

from openmic.tts import TTSConfig, TTSRequestError, normalize_wav_header


SAMPLE_RATE = 16000


def encode_wav(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return buffer.getvalue()


def decode_wav(audio: bytes) -> np.ndarray:
    # Repair streaming placeholders, but do not silently "repair" a genuinely
    # truncated ordinary WAV or absorb a trailing LIST chunk into PCM data.
    if len(audio) < 12 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        raise ValueError("服务未返回 RIFF/WAVE 音频")
    declared = struct.unpack_from("<I", audio, 4)[0]
    placeholder = declared == 0 or declared >= 0x7FFF0000
    if not placeholder and declared + 8 > len(audio):
        raise ValueError("WAV RIFF 数据截断")
    offset = 12
    while offset + 8 <= len(audio):
        size = struct.unpack_from("<I", audio, offset + 4)[0]
        if audio[offset:offset + 4] == b"data":
            if size >= 0x7FFF0000 or (size == 0 and placeholder):
                audio = normalize_wav_header(audio)
                size = struct.unpack_from("<I", audio, offset + 4)[0]
            elif offset + 8 + size > len(audio):
                raise ValueError("WAV PCM 数据截断")
            if size % 2:
                raise ValueError("16-bit PCM 含不完整采样帧")
            break
        if offset + 8 + size > len(audio):
            raise ValueError("WAV chunk 数据截断")
        offset += 8 + size + size % 2
    if placeholder:
        fixed = bytearray(audio)
        struct.pack_into("<I", fixed, 4, len(audio) - 8)
        audio = bytes(fixed)
    try:
        with wave.open(io.BytesIO(audio), "rb") as handle:
            if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth(),
                    handle.getcomptype()) != (SAMPLE_RATE, 1, 2, "NONE"):
                raise ValueError("需要 16000 Hz、单声道、16-bit PCM WAV")
            count = handle.getnframes()
            data = handle.readframes(count)
            if count == 0 or len(data) != count * 2:
                raise ValueError("WAV 为空或音频数据截断")
            return np.frombuffer(data, dtype="<i2").copy()
    except (wave.Error, EOFError) as exc:
        raise ValueError("服务未返回有效的 PCM WAV") from exc


class SiliconFlowBackend:
    synthetic = False

    def __init__(self, config: Optional[TTSConfig] = None, *,
                 voice: Optional[str] = None, session: Optional[Any] = None):
        self.config = replace(config or TTSConfig.from_env(), response_format="wav", sample_rate=16000)
        if self.config.model != "FunAudioLLM/CosyVoice2-0.5B":
            raise ValueError("该实验适配器只验证了 CosyVoice2 指令协议，请设置 TTS_MODEL=FunAudioLLM/CosyVoice2-0.5B")
        self.voice = voice or self.config.voice
        self.session = session or requests.Session()
        self.identity = {"backend": "siliconflow", "model": self.config.model,
                         "voice": self.voice, "endpoint": self.config.base_url,
                         "sample_rate": SAMPLE_RATE, "adapter": 1}

    def synthesize(self, text: str, *, instruction: str, speed: float) -> bytes:
        # Official examples place instruction BEFORE <|endofprompt|>.
        payload = {"model": self.config.model, "voice": self.voice,
                   "input": instruction + "<|endofprompt|>" + text,
                   "response_format": "wav", "sample_rate": SAMPLE_RATE,
                   "speed": speed, "gain": 0, "stream": False}
        try:
            response = self.session.post(
                self.config.base_url.rstrip("/") + "/audio/speech",
                headers={"Authorization": "Bearer " + self.config.api_key,
                         "Content-Type": "application/json"},
                json=payload, timeout=(15, self.config.timeout_seconds))
        except requests.RequestException as exc:
            # Avoid displaying request headers/keys from vendor exceptions.
            raise TTSRequestError("TTS 网络请求失败；已完成的段可从缓存恢复，请检查网络后重试") from None
        if response.status_code != 200:
            raise TTSRequestError(f"TTS HTTP {response.status_code}；检查额度、模型权限和 endpoint 后重试")
        # Validate the original declaration before any placeholder repair.
        return encode_wav(decode_wav(response.content))


class ToneBackend:
    """Deterministic fake audio, for plumbing tests only. Never a TTS score."""

    synthetic = True
    identity = {"backend": "tone", "model": "synthetic-not-speech", "adapter": 1,
                "sample_rate": SAMPLE_RATE}

    def synthesize(self, text: str, *, instruction: str, speed: float) -> bytes:
        seconds = min(8.0, max(0.2, len(text) * 0.075 / speed))
        frequency = 220 + int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:4], 16) % 220
        signal = np.sin(2 * np.pi * frequency * np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE)
        signal = (signal * 3500).astype("<i2")
        return encode_wav(np.pad(signal, (800, 800)))
