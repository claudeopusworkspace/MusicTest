"""
Song arranger — assembles sections into a full-length piece with crossfade transitions.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Section:
    """A section of a song (e.g., 8 bars of layered audio)."""
    name: str
    audio: np.ndarray  # Stereo float32, shape (N, 2)
    sr: int = 44100


def load_section(path: str, sr: int = 44100) -> np.ndarray:
    """Load a mixed section from a WAV file."""
    data, file_sr = sf.read(path, dtype="float32")
    if file_sr != sr:
        import librosa
        if data.ndim > 1:
            channels = []
            for ch in range(data.shape[1]):
                channels.append(librosa.resample(data[:, ch],
                                                 orig_sr=file_sr,
                                                 target_sr=sr))
            data = np.column_stack(channels)
        else:
            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
    if data.ndim == 1:
        data = np.column_stack([data, data])
    return data


def crossfade(a: np.ndarray, b: np.ndarray, fade_samples: int) -> np.ndarray:
    """
    Join two stereo arrays with a crossfade overlap.

    The last `fade_samples` of `a` overlap with the first `fade_samples` of `b`.
    Uses equal-power (cosine) crossfade for smooth transitions.
    """
    if fade_samples <= 0:
        return np.concatenate([a, b], axis=0)

    fade_samples = min(fade_samples, a.shape[0], b.shape[0])

    # Equal-power crossfade curve
    t = np.linspace(0, np.pi / 2, fade_samples)
    fade_out = np.cos(t)[:, np.newaxis]  # (fade_samples, 1) for broadcasting
    fade_in = np.sin(t)[:, np.newaxis]

    # Non-overlapping portions
    a_head = a[:-fade_samples]
    b_tail = b[fade_samples:]

    # Overlapping portion
    overlap = a[-fade_samples:] * fade_out + b[:fade_samples] * fade_in

    return np.concatenate([a_head, overlap, b_tail], axis=0)


def arrange(sections: list, crossfade_seconds: float = 0.5,
            sr: int = 44100) -> np.ndarray:
    """
    Arrange a list of Sections (or stereo arrays) into a full piece.

    Args:
        sections: List of Section objects or stereo numpy arrays
        crossfade_seconds: Duration of crossfade between sections
        sr: Sample rate

    Returns:
        Stereo float32 numpy array of the full arrangement
    """
    fade_samples = int(crossfade_seconds * sr)

    arrays = []
    for s in sections:
        if isinstance(s, Section):
            arrays.append(s.audio)
        else:
            arrays.append(s)

    if not arrays:
        return np.zeros((0, 2), dtype=np.float32)

    result = arrays[0]
    for arr in arrays[1:]:
        result = crossfade(result, arr, fade_samples)

    return result


def make_loopable(audio: np.ndarray, crossfade_seconds: float = 0.5,
                  sr: int = 44100) -> np.ndarray:
    """
    Make an arrangement seamlessly loopable by crossfading the tail into the head.

    Trims the end and blends it back into the beginning so that when the
    audio loops, there's no click or discontinuity.
    """
    fade_samples = int(crossfade_seconds * sr)
    if fade_samples >= audio.shape[0] // 2:
        return audio  # Too short to meaningfully crossfade

    # Equal-power curves
    t = np.linspace(0, np.pi / 2, fade_samples)
    fade_in = np.sin(t)[:, np.newaxis]
    fade_out = np.cos(t)[:, np.newaxis]

    # Blend: the beginning fades in from the tail, the tail fades out into the beginning
    result = audio.copy()
    result[:fade_samples] = (audio[:fade_samples] * fade_in +
                              audio[-fade_samples:] * fade_out)
    # Trim the tail that's now been folded into the head
    result = result[:-fade_samples]

    return result


def export_arrangement(audio: np.ndarray, path: str, sr: int = 44100,
                       normalize: bool = True, headroom_db: float = -1.0) -> str:
    """Save an arranged piece to WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        peak = np.abs(audio).max()
        if peak > 0:
            target = 10 ** (headroom_db / 20)
            audio = audio * (target / peak)

    sf.write(str(path), audio, sr)
    return str(path)
