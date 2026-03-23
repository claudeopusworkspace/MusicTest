"""Quick test: generate a short melody with the synth and analyze it."""

import sys
sys.path.insert(0, "/workspace/MusicTest")

from src.synth import Synth, render_melody_to_bars
from src.analyze import full_analysis, detect_key
import soundfile as sf
import numpy as np
from pathlib import Path

OUTPUT_DIR = "/workspace/MusicTest/experiments/synth_test"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

SR = 44100
BPM = 130

# Test a warm synth — this will be our melody voice
synth = Synth(
    waveform="warm",
    attack=0.03,
    decay=0.15,
    sustain=0.6,
    release=0.25,
    vibrato_rate=4.5,
    vibrato_depth=0.15,
    brightness=0.4,
    detune_cents=4.0,
)

# Simple G major test melody (8 bars)
test_melody = [
    # Bar 1-2: opening phrase
    ("G4", 1.0, 0.8), ("B4", 0.5, 0.7), ("D5", 0.5, 0.75),
    ("E5", 1.0, 0.85), ("D5", 0.5, 0.7), ("B4", 0.5, 0.65),
    # Bar 3-4: answering phrase
    ("A4", 1.0, 0.8), ("B4", 0.5, 0.7), ("G4", 0.5, 0.6),
    ("A4", 1.5, 0.75), ("R", 0.5, 0),
    # Bar 5-6: variation
    ("G4", 0.5, 0.7), ("A4", 0.5, 0.7), ("B4", 1.0, 0.85),
    ("D5", 0.5, 0.8), ("E5", 0.5, 0.75), ("D5", 1.0, 0.8),
    # Bar 7-8: resolution
    ("B4", 0.5, 0.7), ("A4", 0.5, 0.65), ("G4", 1.5, 0.85),
    ("R", 0.5, 0), ("G4", 1.0, 0.6),
    # Pad remainder to fill 8 bars (32 beats total)
]

# Count beats used
total = sum(d for _, d, _ in test_melody)
remaining = 32 - total
if remaining > 0:
    test_melody.append(("R", remaining, 0))

audio = render_melody_to_bars(test_melody, synth, bpm=BPM, bars=8, sr=SR)
path = f"{OUTPUT_DIR}/warm_synth_test.wav"
sf.write(path, audio, SR)

print(f"Generated: {path}")
print(f"Duration: {len(audio)/SR:.2f}s")
print(f"Peak amplitude: {np.abs(audio).max():.3f}")

key = detect_key(path)
print(f"Detected key: {key['full']} (confidence: {key['confidence']:.3f})")

# Also test a triangle wave for comparison
synth_tri = Synth(
    waveform="triangle",
    attack=0.02,
    decay=0.1,
    sustain=0.65,
    release=0.3,
    vibrato_rate=5.0,
    vibrato_depth=0.2,
    detune_cents=5.0,
)

audio_tri = render_melody_to_bars(test_melody, synth_tri, bpm=BPM, bars=8, sr=SR)
path_tri = f"{OUTPUT_DIR}/triangle_synth_test.wav"
sf.write(path_tri, audio_tri, SR)
print(f"\nGenerated triangle: {path_tri}")
key_tri = detect_key(path_tri)
print(f"Detected key: {key_tri['full']} (confidence: {key_tri['confidence']:.3f})")

# Run full analysis on the warm synth
print("\nRunning analysis on warm synth...")
full_analysis(path)
print("Done.")
