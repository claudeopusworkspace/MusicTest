"""
Audio analysis and visualization — generates spectrograms, chromagrams,
and waveform plots that Claude can interpret visually.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def load_audio(path: str, sr: int = 48000, mono: bool = True):
    """Load audio file, return (samples, sample_rate)."""
    y, sr_actual = librosa.load(path, sr=sr, mono=mono)
    return y, sr_actual


def spectrogram(path: str, output_path: str = None, sr: int = 48000,
                title: str = None) -> str:
    """Generate a mel spectrogram image from an audio file."""
    y, sr = load_audio(path, sr)
    if output_path is None:
        output_path = str(Path(path).with_suffix(".spectrogram.png"))
    if title is None:
        title = f"Mel Spectrogram — {Path(path).stem}"

    fig, ax = plt.subplots(figsize=(14, 5))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr // 2)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis="time", y_axis="mel",
                                   sr=sr, fmax=sr // 2, ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def chromagram(path: str, output_path: str = None, sr: int = 48000,
               title: str = None) -> str:
    """Generate a chromagram image showing pitch class distribution over time."""
    y, sr = load_audio(path, sr)
    if output_path is None:
        output_path = str(Path(path).with_suffix(".chromagram.png"))
    if title is None:
        title = f"Chromagram — {Path(path).stem}"

    fig, ax = plt.subplots(figsize=(14, 4))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    img = librosa.display.specshow(chroma, x_axis="time", y_axis="chroma",
                                   sr=sr, ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def waveform(path: str, output_path: str = None, sr: int = 48000,
             title: str = None) -> str:
    """Generate a waveform plot."""
    y, sr = load_audio(path, sr)
    if output_path is None:
        output_path = str(Path(path).with_suffix(".waveform.png"))
    if title is None:
        title = f"Waveform — {Path(path).stem}"

    fig, ax = plt.subplots(figsize=(14, 3))
    times = np.arange(len(y)) / sr
    ax.plot(times, y, linewidth=0.3, color="#2196F3")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(0, times[-1])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def full_analysis(path: str, output_dir: str = None, sr: int = 48000) -> dict:
    """Generate all visualizations for an audio file. Returns dict of paths."""
    p = Path(path)
    if output_dir is None:
        output_dir = str(p.parent)
    stem = p.stem
    out = Path(output_dir)

    return {
        "spectrogram": spectrogram(path, str(out / f"{stem}.spectrogram.png"), sr),
        "chromagram": chromagram(path, str(out / f"{stem}.chromagram.png"), sr),
        "waveform": waveform(path, str(out / f"{stem}.waveform.png"), sr),
    }


def detect_key(path: str, sr: int = 48000) -> dict:
    """Estimate the musical key of an audio file."""
    y, sr = load_audio(path, sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    # Krumhansl-Kessler key profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    note_names = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]

    best_corr = -1
    best_key = "C"
    best_mode = "major"

    for i in range(12):
        rolled = np.roll(chroma_mean, -i)
        for profile, mode in [(major_profile, "major"), (minor_profile, "minor")]:
            corr = np.corrcoef(rolled, profile)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_key = note_names[i]
                best_mode = mode

    return {
        "key": best_key,
        "mode": best_mode,
        "full": f"{best_key} {best_mode}",
        "confidence": float(best_corr),
    }


def detect_tempo(path: str, sr: int = 48000) -> float:
    """Estimate BPM of an audio file."""
    y, sr = load_audio(path, sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)
