"""
Experiment: Can Foundation-1 follow explicit melody instructions in prompts?

Test whether the text encoder understands note-level directions like
"play C4 D4 E4 F4" or "ascending scale" and actually produces those notes
in that order.
"""

import sys
sys.path.insert(0, "/workspace/MusicTest")

from src.generate import generate, save_wav, load_model, bars_bpm_to_seconds
from src.analyze import full_analysis, detect_key

print("Loading model...")
load_model()
print("Model loaded.\n")

TESTS = [
    # Explicit note sequences
    ("piano_CDEF_explicit",
     "Piano, playing C4 D4 E4 F4, quarter notes, C major, 1 Bars, 120 BPM",
     1, 120, 42),

    ("piano_CDEG_explicit",
     "Piano, playing the notes C4 D4 E4 G4 in sequence, C major, 1 Bars, 120 BPM",
     1, 120, 42),

    # Descriptive melodic direction
    ("piano_ascending_scale",
     "Piano, ascending scale from C to G, stepwise motion, C major, 1 Bars, 120 BPM",
     1, 120, 42),

    ("piano_descending_melody",
     "Piano, descending melody from G4 to C4, C major, 1 Bars, 120 BPM",
     1, 120, 42),

    # Interval-based description
    ("guitar_simple_arpeggio",
     "Acoustic guitar, C major arpeggio, C4 E4 G4 C5, C major, 1 Bars, 120 BPM",
     1, 120, 42),

    # Longer sequence (2 bars = 8 beats)
    ("piano_8note_melody",
     "Piano, melody: C4 D4 E4 G4 A4 G4 E4 D4, eighth notes, C major, 2 Bars, 120 BPM",
     2, 120, 42),

    # Try a well-known pattern the model might have seen in training data
    ("piano_twinkle",
     "Piano, simple melody, C C G G A A G, nursery rhyme style, C major, 2 Bars, 120 BPM",
     2, 120, 42),
]

OUTPUT_DIR = "/workspace/MusicTest/experiments/melody_prompts"

for name, prompt, bars, bpm, seed in TESTS:
    print(f"--- Generating: {name} ---")
    print(f"    Prompt: {prompt}")
    print(f"    Duration: {bars_bpm_to_seconds(bars, bpm):.1f}s ({bars} bar @ {bpm} BPM)")

    audio, sr = generate(prompt, bars=bars, bpm=bpm, seed=seed)
    wav_path = f"{OUTPUT_DIR}/{name}.wav"
    save_wav(audio, sr, wav_path)

    key_info = detect_key(wav_path)
    print(f"    Detected key: {key_info['full']} (confidence: {key_info['confidence']:.3f})")

    # Run full analysis
    full_analysis(wav_path)
    print(f"    Analysis complete.")
    print()

print("All melody prompt tests generated.")
