"""
Simple drum pattern sequencer.
Takes a pattern string and drum sample WAVs, places hits at beat positions.
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def load_sample(path: str, target_sr: int = 48000) -> np.ndarray:
    """Load a WAV sample, return as mono float32 numpy array."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # Mix to mono
    if sr != target_sr:
        # Simple resampling via librosa
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data


def beats_to_samples(beat: float, bpm: float, sr: int = 48000) -> int:
    """Convert a beat position to a sample index."""
    seconds_per_beat = 60.0 / bpm
    return int(round(beat * seconds_per_beat * sr))


def sequence_pattern(pattern: str, sample: np.ndarray, bpm: float,
                     total_bars: int, sr: int = 48000,
                     velocity: float = 1.0,
                     subdivisions: int = None) -> np.ndarray:
    """
    Sequence a single drum hit across a pattern.

    Args:
        pattern: String where non-'.' characters trigger the hit.
                 e.g. "x...x...x...x..." for a 4-on-the-floor kick over 1 bar (16th note grid)
        sample: Audio data for the hit (mono float32)
        bpm: Tempo
        total_bars: Total length in bars
        sr: Sample rate
        velocity: Volume multiplier (0.0 - 1.0)
        subdivisions: Grid divisions per bar. Defaults to len(pattern) / total_bars,
                      or len(pattern) if total_bars isn't cleanly divisible.

    Returns:
        Mono float32 numpy array of the full sequenced pattern.
    """
    total_beats = total_bars * 4
    total_samples = beats_to_samples(total_beats, bpm, sr)
    output = np.zeros(total_samples, dtype=np.float32)

    if subdivisions is None:
        subdivisions = len(pattern)

    beats_per_step = total_beats / subdivisions

    for i, char in enumerate(pattern):
        if char != ".":
            # Support velocity via character: x=full, o=medium, -=soft
            vel = velocity
            if char == "o":
                vel *= 0.65
            elif char == "-":
                vel *= 0.35

            beat_pos = i * beats_per_step
            sample_pos = beats_to_samples(beat_pos, bpm, sr)

            if sample_pos < total_samples:
                end = min(sample_pos + len(sample), total_samples)
                length = end - sample_pos
                output[sample_pos:end] += sample[:length] * vel

    return output


def build_drum_track(kit: dict, patterns: dict, bpm: float,
                     total_bars: int, sr: int = 48000) -> np.ndarray:
    """
    Build a complete drum track from a kit and pattern set.

    Args:
        kit: dict mapping names to WAV file paths, e.g. {"kick": "samples/kick.wav"}
        patterns: dict mapping names to pattern strings, e.g. {"kick": "x...x...x...x..."}
        bpm: Tempo
        total_bars: Total length in bars
        sr: Sample rate

    Returns:
        Mixed mono drum track as float32 numpy array.
    """
    total_beats = total_bars * 4
    total_samples = beats_to_samples(total_beats, bpm, sr)
    mix = np.zeros(total_samples, dtype=np.float32)

    for name, pattern in patterns.items():
        if name not in kit:
            raise ValueError(f"No sample in kit for '{name}'")
        sample = load_sample(kit[name], sr)
        track = sequence_pattern(pattern, sample, bpm, total_bars, sr)
        # Ensure same length
        length = min(len(mix), len(track))
        mix[:length] += track[:length]

    return mix


def save_drum_track(track: np.ndarray, path: str, sr: int = 48000):
    """Save a drum track to WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Normalize to prevent clipping
    peak = np.abs(track).max()
    if peak > 1.0:
        track = track / peak
    sf.write(str(path), track, sr)
    return str(path)
