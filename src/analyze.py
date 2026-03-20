"""
Audio analysis and visualization for AI-assisted music composition.

This module generates visual representations of audio that Claude can interpret
by reading the output images. It provides two tiers of analysis:

STANDARD ANALYSIS (full_analysis):
    Run on every generated sample. Produces 7 images that together give a
    comprehensive picture of the audio:

    1. Mel Spectrogram     — Overall frequency content over time. Shows instrument
                             timbre, harmonic structure, and energy distribution.
    2. Chromagram          — Pitch class activity over time. Shows chord changes
                             and harmonic movement as they happen.
    3. Waveform            — Raw amplitude over time. Shows clipping, silence,
                             stereo balance, and gross dynamic shape.
    4. Pitch Histogram     — Total energy per pitch class (bar chart). The most
                             reliable way to identify key — compare the top 3-4
                             notes against known triads/scales.
    5. RMS Energy          — Perceived loudness over time. Reveals rising/falling
                             dynamics, energy buildups, and phrase-level structure.
    6. Spectral Centroid   — Brightness (center frequency) over time. A rising
                             centroid means the sound is getting brighter. Useful
                             for detecting filter sweeps and timbral evolution.
    7. Tempogram           — Tempo estimation over time with BPM readout. Gives
                             a precise tempo number and shows whether it's steady.

DIAGNOSTIC ANALYSIS (called individually when investigating a specific question):
    - zoomed_spectrogram   — High time-resolution view of a short window. Use to
                             investigate reverb tails, transient character, or
                             artifacts. NOT included in full_analysis because a
                             single zoomed window may not represent the whole sample.

INTERPRETATION WORKFLOW:
    1. Generate a sample with src.generate
    2. Run full_analysis() to produce all 7 standard images
    3. Read the images to assess the output
    4. If a specific question remains (e.g. "is there reverb?"), use the
       appropriate diagnostic function for a closer look
    5. Use detect_key() and detect_tempo() for numeric confirmation
"""

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def load_audio(path: str, sr: int = 44100, mono: bool = True):
    """Load audio file, return (samples, sample_rate)."""
    y, sr_actual = librosa.load(path, sr=sr, mono=mono)
    return y, sr_actual


def spectrogram(path: str, output_path: str = None, sr: int = 44100,
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


def chromagram(path: str, output_path: str = None, sr: int = 44100,
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


def waveform(path: str, output_path: str = None, sr: int = 44100,
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


def pitch_class_histogram(path: str, output_path: str = None, sr: int = 44100,
                          title: str = None) -> str:
    """Bar chart of total energy per pitch class — makes key identification unambiguous."""
    y, sr = load_audio(path, sr)
    if output_path is None:
        output_path = str(Path(path).with_suffix(".pitch_histogram.png"))
    if title is None:
        title = f"Pitch Class Energy — {Path(path).stem}"

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_energy = chroma.mean(axis=1)
    note_names = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(note_names, chroma_energy, color="#4CAF50", edgecolor="black",
                  linewidth=0.5)
    # Highlight the top 3 notes
    sorted_idx = np.argsort(chroma_energy)[::-1]
    for rank, idx in enumerate(sorted_idx[:3]):
        bars[idx].set_color(["#F44336", "#FF9800", "#FFC107"][rank])
    ax.set_ylabel("Mean Energy")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def rms_energy(path: str, output_path: str = None, sr: int = 44100,
               title: str = None) -> str:
    """RMS energy envelope over time — shows loudness dynamics and rising/falling character."""
    y, sr = load_audio(path, sr)
    if output_path is None:
        output_path = str(Path(path).with_suffix(".rms_energy.png"))
    if title is None:
        title = f"RMS Energy — {Path(path).stem}"

    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(times, rms, color="#E91E63", linewidth=1.2)
    ax.fill_between(times, rms, alpha=0.3, color="#E91E63")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS Energy")
    ax.set_title(title)
    ax.set_xlim(0, times[-1])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def spectral_centroid(path: str, output_path: str = None, sr: int = 44100,
                      title: str = None) -> str:
    """Spectral centroid over time — tracks brightness/timbral evolution."""
    y, sr = load_audio(path, sr)
    if output_path is None:
        output_path = str(Path(path).with_suffix(".spectral_centroid.png"))
    if title is None:
        title = f"Spectral Centroid — {Path(path).stem}"

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    times = librosa.times_like(cent, sr=sr)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(times, cent, color="#9C27B0", linewidth=1.0)
    ax.fill_between(times, cent, alpha=0.2, color="#9C27B0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.set_xlim(0, times[-1])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def tempogram(path: str, output_path: str = None, sr: int = 44100,
              title: str = None) -> str:
    """Tempogram showing tempo estimates over time — gives precise BPM reading."""
    y, sr = load_audio(path, sr)
    if output_path is None:
        output_path = str(Path(path).with_suffix(".tempogram.png"))
    if title is None:
        title = f"Tempogram — {Path(path).stem}"

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    fig, ax = plt.subplots(figsize=(14, 4))
    librosa.display.specshow(tg, x_axis="time", y_axis="tempo", sr=sr, ax=ax)
    ax.axhline(tempo, color="white", linestyle="--", linewidth=1.5,
               label=f"Estimated: {tempo:.1f} BPM")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def zoomed_spectrogram(path: str, center_time: float = None,
                       window_seconds: float = 1.5, output_path: str = None,
                       sr: int = 44100, title: str = None) -> str:
    """
    High time-resolution spectrogram of a short window.

    DIAGNOSTIC TOOL — not part of the standard analysis suite. Use when
    investigating a specific question about a localized moment in the audio,
    such as:
      - Reverb: look for gradual spectral decay after a transient
      - Transient quality: sharp vs. soft attack
      - Artifacts: clicks, pops, or glitches
      - FX tails: delay repeats, phaser sweeps

    Note: a single zoomed window may not be representative of the whole sample.
    Choose center_time deliberately, or leave it as None to auto-select the
    loudest transient.

    Args:
        path: Audio file path
        center_time: Center of the zoom window in seconds. If None, auto-selects
                     the time of the strongest onset (loudest transient).
        window_seconds: Duration of the zoom window (default 1.5s)
        output_path: Where to save the image
        sr: Sample rate
        title: Plot title
    """
    y, sr = load_audio(path, sr)
    duration = len(y) / sr

    if center_time is None:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_times = librosa.times_like(onset_env, sr=sr)
        center_time = onset_times[np.argmax(onset_env)]

    start_time = max(0, center_time - window_seconds / 2)
    end_time = min(duration, center_time + window_seconds / 2)
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    y_window = y[start_sample:end_sample]

    if output_path is None:
        output_path = str(Path(path).with_suffix(".zoomed_spectrogram.png"))
    if title is None:
        title = (f"Zoomed Spectrogram — {Path(path).stem} "
                 f"[{start_time:.2f}s – {end_time:.2f}s]")

    fig, ax = plt.subplots(figsize=(14, 5))
    S = librosa.feature.melspectrogram(y=y_window, sr=sr, n_mels=128,
                                       fmax=sr // 2, hop_length=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis="time", y_axis="mel",
                                   sr=sr, fmax=sr // 2, ax=ax,
                                   hop_length=128)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def full_analysis(path: str, output_dir: str = None, sr: int = 44100) -> dict:
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
        "pitch_histogram": pitch_class_histogram(path, str(out / f"{stem}.pitch_histogram.png"), sr),
        "rms_energy": rms_energy(path, str(out / f"{stem}.rms_energy.png"), sr),
        "spectral_centroid": spectral_centroid(path, str(out / f"{stem}.spectral_centroid.png"), sr),
        "tempogram": tempogram(path, str(out / f"{stem}.tempogram.png"), sr),
    }


def detect_key(path: str, sr: int = 44100) -> dict:
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


def detect_tempo(path: str, sr: int = 44100) -> float:
    """Estimate BPM of an audio file."""
    y, sr = load_audio(path, sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)
