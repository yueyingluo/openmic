"""Complete waveform decoding and frozen audio features; no quality claims from RMS."""

from math import gcd
from pathlib import Path

import numpy as np

DESCRIPTORS = [
    "duration_seconds",
    "rms_dbfs",
    "peak",
    "low_amplitude_fraction",
    "clipped_fraction",
    "zero_crossing_rate",
    "frame_rms_p10",
    "frame_rms_p50",
    "frame_rms_p90",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "high_frequency_fraction",
    "spectral_flatness_mean",
    "crest_factor",
]


def load_waveform(path):
    import soundfile as sf
    from scipy.signal import resample_poly

    info = sf.info(path)
    wave, sr = sf.read(path, dtype="float32", always_2d=True)
    if not len(wave) or not np.isfinite(wave).all() or len(wave) != info.frames:
        raise ValueError("Incomplete, empty, or nonfinite waveform")
    frames, channels = wave.shape
    mono = wave.mean(1)
    if sr != 16000:
        divisor = gcd(sr, 16000)
        mono = resample_poly(mono, 16000 // divisor, sr // divisor).astype("float32")
    return mono, {
        "source_frames": frames,
        "source_sample_rate": sr,
        "source_channels": channels,
        "duration_seconds": frames / sr,
        "decoded_fraction": 1.0,
        "resampled_frames": len(mono),
        "mixdown": "arithmetic mean of all channels",
        "sample_rate": 16000,
    }


def waveform_descriptors(wave):
    from scipy.signal import stft

    wave = np.asarray(wave, dtype=np.float64)
    if len(wave) < 400:
        raise ValueError("Audio shorter than 25 ms")
    rms = np.sqrt(np.mean(wave**2))
    peak = np.max(np.abs(wave))
    # Include the final partial frame through zero padding, never cut the tail.
    padded = np.pad(wave, (0, (-len(wave)) % 400))
    frames = padded.reshape(-1, 400)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    f, _, z = stft(
        wave, fs=16000, nperseg=400, noverlap=240, boundary="zeros", padded=True
    )
    power = np.abs(z) ** 2 + 1e-12
    total = power.sum(0)
    centroid = (f[:, None] * power).sum(0) / total
    values = [
        len(wave) / 16000,
        20 * np.log10(max(rms, 1e-12)),
        peak,
        np.mean(np.abs(wave) < 0.01),
        np.mean(np.abs(wave) >= 0.999),
        np.mean(np.diff(wave >= 0) != 0),
        *np.quantile(frame_rms, [0.1, 0.5, 0.9]),
        centroid.mean(),
        centroid.std(),
        power[f >= 4000].sum() / power.sum(),
        np.mean(np.exp(np.log(power).mean(0)) / power.mean(0)),
        peak / max(rms, 1e-12),
    ]
    return np.asarray(values, dtype=np.float32)


def load_audio_encoder(model_path, device="cuda", standalone=False):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoder,
    )

    if standalone:
        return (
            Qwen2_5OmniAudioEncoder.from_pretrained(model_path, local_files_only=True)
            .to(device)
            .eval()
        )
    config = AutoConfig.from_pretrained(
        model_path, local_files_only=True
    ).thinker_config.audio_config
    config._attn_implementation = "sdpa"
    tower = Qwen2_5OmniAudioEncoder(config)
    prefix = "thinker.audio_tower."
    # Only this pinned checkpoint's audio tower, never initialize missing weights silently.
    state = {}
    for shard in Path(model_path).glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as source:
            for key in source.keys():
                if key.startswith(prefix):
                    state[key[len(prefix) :]] = source.get_tensor(key)
    tower.load_state_dict(state, strict=True)
    return tower.to(device=device, dtype=torch.float32).eval()


def audio_embeddings(waves, extractor, encoder, device="cuda"):
    import torch

    inputs = extractor(
        waves,
        sampling_rate=16000,
        padding=True,
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )
    mask = (
        inputs["attention_mask"]
        if "attention_mask" in inputs
        else inputs["feature_attention_mask"]
    )
    lengths = mask.sum(1).to(device)
    for wave, n in zip(waves, lengths.tolist()):
        if abs(n / 100 - len(wave) / 16000) > 0.1:
            raise ValueError("Audio feature coverage differs from complete waveform")
    features = inputs["input_features"].to(device)
    features = features.permute(0, 2, 1)[mask.to(device).bool()].permute(1, 0)
    cnn_lengths, out_lengths = encoder._get_feat_extract_output_lengths(lengths)
    with torch.inference_mode():
        with torch.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"
        ):
            encoded = encoder(
                features, feature_lens=lengths, aftercnn_lens=cnn_lengths
            ).last_hidden_state.float()
    if len(encoded) != int(out_lengths.sum()):
        raise ValueError("Encoder length mismatch")
    chunks = encoded.split(out_lengths.tolist())
    vectors = torch.stack(
        [torch.cat([x.mean(0), x.std(0, unbiased=False)]) for x in chunks]
    )
    return vectors.cpu().numpy(), lengths.cpu().tolist()
