"""
Simple synthesizer for programmatic melody composition.

Generates individual notes with controllable pitch, duration, timbre,
and dynamics. Designed to work alongside Foundation-1 layers — provides
note-level melodic control that the AI model can't offer.

Usage:
    from src.synth import Synth, render_melody

    synth = Synth(waveform="triangle", attack=0.05, release=0.3)
    melody = [
        ("C4", 0.5, 0.8),   # (note, duration_beats, velocity)
        ("D4", 0.5, 0.7),
        ("E4", 1.0, 0.9),
    ]
    audio = render_melody(melody, synth, bpm=130, sr=44100)
"""

import numpy as np
from dataclasses import dataclass

# MIDI note number mapping
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_ENHARMONIC = {"Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#", "Ab": "G#",
               "Bb": "A#", "Cb": "B", "E#": "F", "B#": "C"}


def note_to_midi(note: str) -> int:
    """Convert note name (e.g. 'C4', 'F#3', 'Bb5') to MIDI number."""
    # Parse note name and octave
    if len(note) >= 3 and note[1] in ("#", "b"):
        name, octave = note[:2], int(note[2:])
    else:
        name, octave = note[0], int(note[1:])

    name = _ENHARMONIC.get(name, name)
    idx = _NOTE_NAMES.index(name)
    return (octave + 1) * 12 + idx


def midi_to_freq(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def note_to_freq(note: str) -> float:
    """Convert note name to frequency."""
    return midi_to_freq(note_to_midi(note))


@dataclass
class Synth:
    """A simple synthesizer voice with ADSR envelope and basic timbre control."""

    waveform: str = "triangle"   # sine, triangle, saw, square, warm
    attack: float = 0.02         # seconds
    decay: float = 0.1           # seconds
    sustain: float = 0.7         # level (0-1)
    release: float = 0.2         # seconds
    vibrato_rate: float = 0.0    # Hz (0 = off)
    vibrato_depth: float = 0.0   # semitones
    brightness: float = 0.5      # 0-1, controls harmonic content for 'warm' waveform
    detune_cents: float = 3.0    # slight detuning for richness (0 = off)

    def _oscillator(self, freq: float, duration: float, sr: int) -> np.ndarray:
        """Generate raw oscillator waveform."""
        t = np.arange(int(duration * sr)) / sr

        # Apply vibrato
        if self.vibrato_rate > 0 and self.vibrato_depth > 0:
            vibrato = self.vibrato_depth * np.sin(2 * np.pi * self.vibrato_rate * t)
            freq_mod = freq * (2.0 ** (vibrato / 12.0))
            phase = 2 * np.pi * np.cumsum(freq_mod) / sr
        else:
            phase = 2 * np.pi * freq * t

        if self.waveform == "sine":
            out = np.sin(phase)
        elif self.waveform == "triangle":
            out = 2 * np.abs(2 * (phase / (2 * np.pi) % 1) - 1) - 1
        elif self.waveform == "saw":
            out = 2 * (phase / (2 * np.pi) % 1) - 1
        elif self.waveform == "square":
            out = np.sign(np.sin(phase))
        elif self.waveform == "warm":
            # Sine + controlled harmonics for a warm, musical tone
            out = np.sin(phase)
            out += self.brightness * 0.5 * np.sin(2 * phase)    # 2nd harmonic
            out += self.brightness * 0.25 * np.sin(3 * phase)   # 3rd
            out += self.brightness * 0.12 * np.sin(4 * phase)   # 4th
            out += self.brightness * 0.06 * np.sin(5 * phase)   # 5th
            out /= (1 + self.brightness * 0.93)  # normalize
        else:
            raise ValueError(f"Unknown waveform: {self.waveform}")

        # Add slight detuning for richness
        if self.detune_cents > 0:
            detune_ratio = 2.0 ** (self.detune_cents / 1200.0)
            if self.vibrato_rate > 0 and self.vibrato_depth > 0:
                phase2 = 2 * np.pi * np.cumsum(freq_mod * detune_ratio) / sr
            else:
                phase2 = 2 * np.pi * freq * detune_ratio * t

            if self.waveform == "warm":
                out2 = np.sin(phase2)
                out2 += self.brightness * 0.5 * np.sin(2 * phase2)
                out2 += self.brightness * 0.25 * np.sin(3 * phase2)
                out2 /= (1 + self.brightness * 0.75)
            elif self.waveform == "sine":
                out2 = np.sin(phase2)
            elif self.waveform == "triangle":
                out2 = 2 * np.abs(2 * (phase2 / (2 * np.pi) % 1) - 1) - 1
            elif self.waveform == "saw":
                out2 = 2 * (phase2 / (2 * np.pi) % 1) - 1
            else:
                out2 = np.sign(np.sin(phase2))

            out = 0.5 * (out + out2)

        return out.astype(np.float32)

    def _envelope(self, duration: float, sr: int) -> np.ndarray:
        """Generate ADSR envelope."""
        total_samples = int(duration * sr)
        env = np.zeros(total_samples, dtype=np.float32)

        attack_samples = int(self.attack * sr)
        decay_samples = int(self.decay * sr)
        release_samples = int(self.release * sr)

        # Sustain fills whatever is left
        sustain_samples = max(0, total_samples - attack_samples
                              - decay_samples - release_samples)

        pos = 0

        # Attack
        n = min(attack_samples, total_samples - pos)
        if n > 0:
            env[pos:pos + n] = np.linspace(0, 1, n)
            pos += n

        # Decay
        n = min(decay_samples, total_samples - pos)
        if n > 0:
            env[pos:pos + n] = np.linspace(1, self.sustain, n)
            pos += n

        # Sustain
        n = min(sustain_samples, total_samples - pos)
        if n > 0:
            env[pos:pos + n] = self.sustain
            pos += n

        # Release
        n = total_samples - pos
        if n > 0:
            env[pos:pos + n] = np.linspace(self.sustain, 0, n)

        return env

    def render_note(self, note: str, duration: float, velocity: float = 1.0,
                    sr: int = 44100) -> np.ndarray:
        """
        Render a single note.

        Args:
            note: Note name (e.g. 'C4', 'G#3')
            duration: Duration in seconds
            velocity: Volume (0-1)
            sr: Sample rate

        Returns:
            Mono float32 numpy array
        """
        freq = note_to_freq(note)
        wave = self._oscillator(freq, duration, sr)
        env = self._envelope(duration, sr)
        return wave * env * velocity


def render_melody(notes: list, synth: Synth, bpm: float = 120,
                  sr: int = 44100) -> np.ndarray:
    """
    Render a melody from a list of note events.

    Args:
        notes: List of (note_name, duration_beats, velocity) tuples.
               Use "R" or "rest" as note_name for rests.
        synth: Synth instance to use for rendering
        bpm: Tempo in beats per minute
        sr: Sample rate

    Returns:
        Mono float32 numpy array of the complete melody
    """
    seconds_per_beat = 60.0 / bpm

    # Calculate total length
    total_beats = sum(dur for _, dur, _ in notes)
    total_samples = int(total_beats * seconds_per_beat * sr)
    output = np.zeros(total_samples, dtype=np.float32)

    pos = 0  # current position in samples
    for note_name, duration_beats, velocity in notes:
        duration_seconds = duration_beats * seconds_per_beat
        duration_samples = int(duration_seconds * sr)

        if note_name.upper() not in ("R", "REST"):
            rendered = synth.render_note(note_name, duration_seconds,
                                         velocity, sr)
            end = min(pos + len(rendered), total_samples)
            output[pos:end] += rendered[:end - pos]

        pos += duration_samples

    return output


def render_melody_to_bars(notes: list, synth: Synth, bpm: float = 120,
                          bars: int = 8, sr: int = 44100) -> np.ndarray:
    """
    Render a melody, trimmed/padded to an exact bar count.
    Ensures the output aligns perfectly with other bar-based layers.
    """
    seconds_per_beat = 60.0 / bpm
    target_samples = int(bars * 4 * seconds_per_beat * sr)

    raw = render_melody(notes, synth, bpm, sr)

    if len(raw) >= target_samples:
        return raw[:target_samples]
    else:
        padded = np.zeros(target_samples, dtype=np.float32)
        padded[:len(raw)] = raw
        return padded
