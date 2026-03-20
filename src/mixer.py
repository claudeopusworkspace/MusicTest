"""
Multitrack audio mixer.
Layers multiple audio files/arrays with volume, pan, and timing control.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Track:
    """A single track in the mix."""
    name: str
    audio: np.ndarray  # Mono or stereo float32
    sr: int = 48000
    volume: float = 1.0  # Linear gain (0.0 - 2.0+)
    pan: float = 0.0  # -1.0 (left) to 1.0 (right)
    offset_samples: int = 0  # Start position in the mix


class Mixer:
    """Simple multitrack mixer that outputs stereo WAV."""

    def __init__(self, sr: int = 48000):
        self.sr = sr
        self.tracks: list[Track] = []

    def add_track(self, name: str, audio, volume: float = 1.0,
                  pan: float = 0.0, offset_seconds: float = 0.0) -> Track:
        """
        Add a track to the mix.

        Args:
            name: Track identifier
            audio: File path (str/Path) or numpy array
            volume: Linear gain
            pan: Stereo position (-1 to 1)
            offset_seconds: When this track starts in the mix
        """
        if isinstance(audio, (str, Path)):
            data, file_sr = sf.read(str(audio), dtype="float32")
            if file_sr != self.sr:
                import librosa
                if data.ndim > 1:
                    # Resample each channel
                    channels = []
                    for ch in range(data.shape[1]):
                        channels.append(librosa.resample(data[:, ch],
                                                         orig_sr=file_sr,
                                                         target_sr=self.sr))
                    data = np.column_stack(channels)
                else:
                    data = librosa.resample(data, orig_sr=file_sr,
                                            target_sr=self.sr)
        else:
            data = np.array(audio, dtype=np.float32)

        offset_samples = int(round(offset_seconds * self.sr))

        track = Track(
            name=name,
            audio=data,
            sr=self.sr,
            volume=volume,
            pan=pan,
            offset_samples=offset_samples,
        )
        self.tracks.append(track)
        return track

    def _to_stereo(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio is stereo (N, 2)."""
        if audio.ndim == 1:
            return np.column_stack([audio, audio])
        if audio.shape[1] == 1:
            return np.column_stack([audio[:, 0], audio[:, 0]])
        return audio[:, :2]

    def _apply_pan(self, stereo: np.ndarray, pan: float) -> np.ndarray:
        """Apply constant-power panning."""
        # pan: -1 (left) to 1 (right)
        angle = (pan + 1) / 2 * (np.pi / 2)
        left_gain = np.cos(angle)
        right_gain = np.sin(angle)
        stereo[:, 0] *= left_gain
        stereo[:, 1] *= right_gain
        return stereo

    def mix(self, normalize: bool = True, headroom_db: float = -1.0) -> np.ndarray:
        """
        Mix all tracks down to stereo.

        Returns:
            Stereo float32 numpy array of shape (N, 2)
        """
        if not self.tracks:
            return np.zeros((0, 2), dtype=np.float32)

        # Determine total length
        total_length = 0
        for t in self.tracks:
            track_end = t.offset_samples + (len(t.audio) if t.audio.ndim == 1
                                            else t.audio.shape[0])
            total_length = max(total_length, track_end)

        output = np.zeros((total_length, 2), dtype=np.float32)

        for t in self.tracks:
            stereo = self._to_stereo(t.audio)
            stereo = stereo * t.volume
            stereo = self._apply_pan(stereo, t.pan)

            start = t.offset_samples
            end = start + stereo.shape[0]
            if end > total_length:
                stereo = stereo[:total_length - start]
                end = total_length
            output[start:end] += stereo

        if normalize:
            peak = np.abs(output).max()
            if peak > 0:
                target = 10 ** (headroom_db / 20)
                output = output * (target / peak)

        return output

    def export(self, path: str, normalize: bool = True,
               headroom_db: float = -1.0) -> str:
        """Mix and save to WAV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mixed = self.mix(normalize=normalize, headroom_db=headroom_db)
        sf.write(str(path), mixed, self.sr)
        return str(path)
