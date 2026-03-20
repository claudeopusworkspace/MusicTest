"""
Programmatic wrapper around Foundation-1 for sample generation.
Bypasses Gradio UI entirely — loads model, builds conditioning, generates audio.
"""

import json
import torch
import torchaudio
from pathlib import Path
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

BEATS_PER_BAR = 4
MODEL_ID = "RoyalCities/Foundation-1"

_model = None
_model_config = None


def load_model(model_id: str = MODEL_ID, device: str = "cuda"):
    """Load Foundation-1 (or another model) and cache it."""
    global _model, _model_config
    if _model is not None:
        return _model, _model_config
    model, config = get_pretrained_model(model_id)
    model = model.to(device).eval().requires_grad_(False)
    _model = model
    _model_config = config
    return model, config


def bars_bpm_to_seconds(bars: int, bpm: float) -> float:
    """Convert bars + BPM to duration in seconds."""
    return (60.0 / bpm) * BEATS_PER_BAR * bars


def bars_bpm_to_samples(bars: int, bpm: float, sample_rate: int = 48000) -> int:
    """Convert bars + BPM to sample count, aligned to model requirements."""
    seconds = bars_bpm_to_seconds(bars, bpm)
    return int(round(seconds * sample_rate))


def build_prompt(instrument: str, timbre: str = "", behavior: str = "",
                 fx: str = "", key: str = "C minor", bars: int = 4,
                 bpm: int = 120) -> str:
    """Build a structured Foundation-1 prompt from components."""
    parts = [p for p in [instrument, timbre, behavior, fx] if p]
    parts.append(f"{key}, {bars} Bars, {bpm} BPM")
    return ", ".join(parts)


def generate(prompt: str, bars: int = 4, bpm: int = 120,
             steps: int = 100, cfg_scale: float = 7.0,
             seed: int = -1, device: str = "cuda") -> tuple:
    """
    Generate an audio sample from a text prompt.

    Returns:
        (audio_tensor, sample_rate) — tensor is [channels, samples], float32
    """
    model, config = load_model(device=device)
    sample_rate = config.get("sample_rate", 48000)

    duration_seconds = bars_bpm_to_seconds(bars, bpm)
    sample_count = bars_bpm_to_samples(bars, bpm, sample_rate)

    # Align to model's min_input_length if present
    if hasattr(model, "min_input_length") and model.min_input_length:
        mil = model.min_input_length
        sample_count = ((sample_count + mil - 1) // mil) * mil

    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": duration_seconds,
    }]

    audio = generate_diffusion_cond(
        model,
        conditioning=conditioning,
        steps=steps,
        cfg_scale=cfg_scale,
        sample_size=sample_count,
        seed=seed,
        device=device,
    )

    # Output shape: [batch, channels, samples] — squeeze batch dim
    audio = audio.squeeze(0).cpu()

    # Trim to exact duration
    exact_samples = int(round(duration_seconds * sample_rate))
    audio = audio[:, :exact_samples]

    return audio, sample_rate


def save_wav(audio: torch.Tensor, sample_rate: int, path: str):
    """Save audio tensor to WAV file. Audio should be float32 [-1, 1]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Clamp to prevent clipping artifacts
    audio = audio.clamp(-1.0, 1.0)
    torchaudio.save(str(path), audio, sample_rate)
    return path


def generate_and_save(prompt: str, output_path: str, bars: int = 4,
                      bpm: int = 120, steps: int = 100,
                      cfg_scale: float = 7.0, seed: int = -1,
                      device: str = "cuda") -> dict:
    """Generate a sample and save it. Returns metadata dict."""
    audio, sr = generate(prompt, bars, bpm, steps, cfg_scale, seed, device)
    path = save_wav(audio, sr, output_path)
    return {
        "path": str(path),
        "prompt": prompt,
        "bars": bars,
        "bpm": bpm,
        "duration_seconds": bars_bpm_to_seconds(bars, bpm),
        "sample_rate": sr,
        "seed": seed,
        "steps": steps,
        "cfg_scale": cfg_scale,
    }
