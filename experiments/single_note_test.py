"""
Experiment: Can Foundation-1 generate clean single notes?

Test various instruments and prompt styles to see if we get isolated,
usable single tones suitable for note-by-note melodic sequencing.
"""

import sys
sys.path.insert(0, "/workspace/MusicTest")

from src.generate import generate, save_wav, load_model, bars_bpm_to_seconds
from src.analyze import full_analysis, detect_key

# Pre-load model once
print("Loading model...")
load_model()
print("Model loaded.\n")

# Test matrix: different instruments and prompt phrasings
# Using 1 bar at 120 BPM = 2 seconds — reasonable for a sustained note
# Using seed=42 for reproducibility
TESTS = [
    # (name, prompt, bars, bpm, seed)
    # Piano
    ("piano_sustained_C4",
     "Piano, single sustained C4 note, C major, 1 Bars, 120 BPM",
     1, 120, 42),
    ("piano_single_note",
     "Piano, one note, sustained, warm tone, C major, 1 Bars, 120 BPM",
     1, 120, 42),

    # Acoustic guitar
    ("guitar_plucked_C",
     "Acoustic guitar, single plucked C note, C major, 1 Bars, 120 BPM",
     1, 120, 42),

    # Violin
    ("violin_sustained_C",
     "Violin, single sustained C note, legato, C major, 1 Bars, 120 BPM",
     1, 120, 42),

    # Synth pad
    ("synth_pad_C",
     "Synth pad, single sustained C note, warm, C major, 1 Bars, 120 BPM",
     1, 120, 42),

    # Try a longer duration for more sustain (2 bars = 4 seconds)
    ("piano_sustained_C4_long",
     "Piano, single sustained C4 note, C major, 2 Bars, 120 BPM",
     2, 120, 42),

    # Try without the bars/BPM context in the prompt body
    # (still need bars/bpm for duration calc, but prompt focuses on the note)
    ("piano_minimal_prompt",
     "A single piano note, middle C, sustained",
     1, 120, 42),
]

OUTPUT_DIR = "/workspace/MusicTest/experiments/single_notes"

for name, prompt, bars, bpm, seed in TESTS:
    print(f"--- Generating: {name} ---")
    print(f"    Prompt: {prompt}")
    print(f"    Duration: {bars_bpm_to_seconds(bars, bpm):.1f}s ({bars} bar @ {bpm} BPM)")

    audio, sr = generate(prompt, bars=bars, bpm=bpm, seed=seed)
    wav_path = f"{OUTPUT_DIR}/{name}.wav"
    save_wav(audio, sr, wav_path)
    print(f"    Saved: {wav_path}")

    # Quick key detection to see if it's hitting the right note
    key_info = detect_key(wav_path)
    print(f"    Detected key: {key_info['full']} (confidence: {key_info['confidence']:.3f})")
    print()

print("All test notes generated. Run analysis next.")
