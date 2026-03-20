"""
Programmatic wrapper around Foundation-1 for sample generation.
Bypasses Gradio UI entirely — loads model, builds conditioning, generates audio.
"""

import json
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond

BEATS_PER_BAR = 4
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models" / "Foundation-1"

_model = None
_model_config = None


def load_model(model_dir: str = None, device: str = "cuda"):
    """Load Foundation-1 from local files and cache it."""
    global _model, _model_config
    if _model is not None:
        return _model, _model_config

    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    config_path = model_dir / "model_config.json"
    weights_path = model_dir / "Foundation_1.safetensors"

    with open(config_path) as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)
    model.load_state_dict(load_ckpt_state_dict(str(weights_path)))
    model = model.to(device).eval().requires_grad_(False)

    _model = model
    _model_config = model_config
    return model, model_config


def bars_bpm_to_seconds(bars: int, bpm: float) -> float:
    """Convert bars + BPM to duration in seconds."""
    return (60.0 / bpm) * BEATS_PER_BAR * bars


def bars_bpm_to_samples(bars: int, bpm: float, sample_rate: int = 44100) -> int:
    """Convert bars + BPM to sample count."""
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
    sample_rate = config.get("sample_rate", 44100)

    duration_seconds = bars_bpm_to_seconds(bars, bpm)
    sample_count = bars_bpm_to_samples(bars, bpm, sample_rate)

    # Align to pretransform downsampling ratio
    downsampling_ratio = config.get("model", {}).get("pretransform", {}).get(
        "config", {}).get("downsampling_ratio", 2048)
    if sample_count % downsampling_ratio != 0:
        sample_count = ((sample_count + downsampling_ratio - 1)
                        // downsampling_ratio) * downsampling_ratio

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
    # Convert from [channels, samples] to [samples, channels] for soundfile
    audio_np = audio.numpy().T
    sf.write(str(path), audio_np, sample_rate)
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
