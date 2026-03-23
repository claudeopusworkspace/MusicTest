"""
Café Day Theme v4 — Synth melody replaces Foundation-1 guitar melodies.

Based on v3 reconstruction (cafe_v3_reconstruct.py) with one key change:
the Foundation-1 guitar layers (guitar_a, guitar_b) are REPLACED by
programmatic synth melodies. This solves the melody consistency problem
where independently-generated guitar_a and guitar_b had disconnected
melodic personalities.

Foundation-1 layers kept: bass, piano, pad (harmonic/textural accompaniment)
Sequenced: drums (from café percussion samples)
New: synth melody with coherent A/B phrases sharing motivic material

Key: G major | BPM: 100 | 8 bars per section (19.2s) | 6 sections
Structure: A1(sparse) → A2(build) → B1(full) → A3(breathe) → B2(full) → A4(resolve)
"""

import sys
sys.path.insert(0, "/workspace/MusicTest")

import numpy as np
import soundfile as sf
from pathlib import Path
from src.synth import Synth, render_melody_to_bars
from src.mixer import Mixer
from src.arrange import arrange, export_arrangement

# === Constants ===
SR = 44100
BPM = 100
BARS = 8

GEN = "generations"
SAMPLES = "samples/cafe_percussion"
OUTPUT = "output"

# === Layer paths ===
# Foundation-1 generated layers (kept as accompaniment)
BASS    = f"{GEN}/cafe_day_bass.wav"
PIANO_A = f"{GEN}/cafe_day_piano.wav"       # seed s77
PIANO_B = f"{GEN}/cafe_day_piano_b.wav"     # seed s22
PAD     = f"{GEN}/cafe_day_pad.wav"

# Pre-built drum variations (from café percussion samples)
DRUMS_SPARSE    = f"{GEN}/cafe_drums_sparse.wav"
DRUMS_BUILDING  = f"{GEN}/cafe_drums_building.wav"
DRUMS_FULL      = f"{GEN}/cafe_drums_full.wav"
DRUMS_BREATHING = f"{GEN}/cafe_drums_breathing.wav"


# === Synth Melody Voice ===
# Warm tone that blends with the Foundation-1 acoustic layers.
# Moderate brightness to sit in the mix, not dominate it.
lead_synth = Synth(
    waveform="warm",
    attack=0.04,
    decay=0.15,
    sustain=0.55,
    release=0.35,
    vibrato_rate=4.0,
    vibrato_depth=0.12,
    brightness=0.3,
    detune_cents=4.0,
)


# === Melody Composition ===
# Key: G major (G A B C D E F#)
# Core motif: G4 → B4 → A4 (rising third, step down)
# A melody: develops motif into relaxed, descending phrases
# B melody: develops motif upward with more rhythmic energy
# Both resolve to G, maintaining coherence across sections.
#
# Beat values at 100 BPM: quarter note = 0.6s, half = 1.2s, whole = 2.4s

melody_a = [
    # Phrase 1 (bars 1-2): core motif → gentle descent
    ("G4",  1.0, 0.75), ("B4",  0.5, 0.7),  ("A4",  0.5, 0.65),
    ("B4",  1.0, 0.8),  ("G4",  0.5, 0.65), ("R",   0.5, 0),
    ("A4",  0.75, 0.7), ("G4",  0.75, 0.65), ("E4",  0.5, 0.6),
    ("R",   1.0, 0),

    # Phrase 2 (bars 3-4): answering phrase — rises then settles
    ("D4",  0.5, 0.65), ("E4",  0.5, 0.7),  ("G4",  1.0, 0.75),
    ("A4",  0.5, 0.7),  ("B4",  1.0, 0.8),  ("A4",  0.5, 0.65),
    ("G4",  1.5, 0.75), ("R",   0.5, 0),
    ("R",   1.0, 0),

    # Phrase 3 (bars 5-6): motif restated with upward reach
    ("G4",  0.5, 0.7),  ("B4",  1.0, 0.8),  ("A4",  0.5, 0.7),
    ("D5",  1.0, 0.85), ("B4",  0.5, 0.7),  ("A4",  0.5, 0.65),
    ("G4",  1.0, 0.75), ("R",   1.0, 0),
    ("R",   1.0, 0),

    # Phrase 4 (bars 7-8): resolution, long tones
    ("E4",  0.5, 0.6),  ("G4",  0.5, 0.65), ("A4",  1.0, 0.7),
    ("G4",  2.0, 0.75),
    ("R",   2.0, 0),
]

melody_b = [
    # Phrase 1 (bars 1-2): motif in higher register, more momentum
    ("B4",  0.5, 0.8),  ("D5",  0.5, 0.85), ("E5",  1.0, 0.9),
    ("D5",  0.5, 0.8),  ("B4",  0.5, 0.75),
    ("D5",  0.5, 0.8),  ("E5",  0.5, 0.85), ("G5",  1.0, 0.9),
    ("E5",  0.5, 0.8),  ("D5",  0.5, 0.75),
    ("R",   1.0, 0),

    # Phrase 2 (bars 3-4): cascading descent back toward home
    ("G5",  0.5, 0.85), ("E5",  0.5, 0.8),  ("D5",  0.5, 0.8),
    ("B4",  1.0, 0.8),  ("A4",  0.5, 0.7),
    ("B4",  0.5, 0.75), ("D5",  1.0, 0.8),  ("B4",  0.5, 0.7),
    ("A4",  1.0, 0.7),
    ("R",   1.0, 0),

    # Phrase 3 (bars 5-6): call-and-response, building
    ("G4",  0.5, 0.7),  ("B4",  0.5, 0.8),  ("D5",  1.0, 0.85),
    ("R",   0.5, 0),
    ("A4",  0.5, 0.75), ("D5",  0.5, 0.8),  ("E5",  1.0, 0.9),
    ("R",   0.5, 0),
    ("B4",  0.5, 0.75), ("D5",  0.5, 0.8),  ("G5",  1.0, 0.9),
    ("R",   0.5, 0),

    # Phrase 4 (bars 7-8): resolution echoing A's ending shape
    ("E5",  0.5, 0.8),  ("D5",  0.5, 0.75), ("B4",  1.0, 0.8),
    ("A4",  0.5, 0.7),  ("G4",  2.0, 0.8),
    ("R",   1.5, 0),
]

# Pad to exactly 32 beats
def pad_to_beats(melody, target=32):
    total = sum(d for _, d, _ in melody)
    remaining = target - total
    if remaining > 0.01:
        melody.append(("R", remaining, 0))
    return melody

melody_a = pad_to_beats(melody_a)
melody_b = pad_to_beats(melody_b)

beats_a = sum(d for _, d, _ in melody_a)
beats_b = sum(d for _, d, _ in melody_b)
print(f"Melody A: {beats_a} beats | Melody B: {beats_b} beats")

# === Render Melodies ===
print("Rendering synth melodies...")
audio_melody_a = render_melody_to_bars(melody_a, lead_synth, bpm=BPM, bars=BARS, sr=SR)
audio_melody_b = render_melody_to_bars(melody_b, lead_synth, bpm=BPM, bars=BARS, sr=SR)

# Save standalone melodies for reference
Path(f"{OUTPUT}/cafe_v4_parts").mkdir(parents=True, exist_ok=True)
sf.write(f"{OUTPUT}/cafe_v4_parts/melody_a.wav", audio_melody_a, SR)
sf.write(f"{OUTPUT}/cafe_v4_parts/melody_b.wav", audio_melody_b, SR)
print("  Done.")


# === Mix Sections ===
# Volume reference from v3 reconstruction (least-squares fit):
#   guitar was 0.78-0.90 (lead voice)
#   bass was 0.47-0.58
#   piano was 0.41-0.58
#   pad was 0.16-0.31
#   drums were 0.32-0.53
#
# The synth melody REPLACES guitar — it takes guitar's role as lead voice
# but at a lower volume since it's more present/synthetic than a generated guitar.
# Guitar volume was ~0.84; synth melody should sit around 0.50-0.60 to blend.

print("Mixing sections...")

def mix_section(layers_config):
    """Mix a section from (name, audio_or_path, volume) tuples."""
    mixer = Mixer(sr=SR)
    for name, audio, volume in layers_config:
        mixer.add_track(name, audio, volume=volume, pan=0.0)
    return mixer.mix(normalize=True, headroom_db=-1.0)


# S1: Sparse — melody + bass + sparse drums (intimate opening)
section_1 = mix_section([
    ("melody", audio_melody_a,  0.55),
    ("bass",   BASS,            0.47),
    ("drums",  DRUMS_SPARSE,    0.37),
])

# S2: Building — add piano + pad + fuller drums
section_2 = mix_section([
    ("melody", audio_melody_a,  0.55),
    ("bass",   BASS,            0.53),
    ("piano",  PIANO_A,         0.41),
    ("pad",    PAD,             0.16),
    ("drums",  DRUMS_BUILDING,  0.47),
])

# S3: B full — melody B + piano_b + pad + full drums (peak energy)
section_3 = mix_section([
    ("melody", audio_melody_b,  0.55),
    ("bass",   BASS,            0.58),
    ("piano",  PIANO_B,         0.58),
    ("pad",    PAD,             0.26),
    ("drums",  DRUMS_FULL,      0.53),
])

# S4: Breathing — melody A + bass only + minimal drums
section_4 = mix_section([
    ("melody", audio_melody_a,  0.50),
    ("bass",   BASS,            0.53),
    ("drums",  DRUMS_BREATHING, 0.32),
])

# S5: B full (repeat of S3 with slight pad increase)
section_5 = mix_section([
    ("melody", audio_melody_b,  0.55),
    ("bass",   BASS,            0.58),
    ("piano",  PIANO_B,         0.58),
    ("pad",    PAD,             0.31),
    ("drums",  DRUMS_FULL,      0.53),
])

# S6: Resolution — melody A + piano + pad + building drums (warm close)
section_6 = mix_section([
    ("melody", audio_melody_a,  0.55),
    ("bass",   BASS,            0.58),
    ("piano",  PIANO_A,         0.46),
    ("pad",    PAD,             0.26),
    ("drums",  DRUMS_BUILDING,  0.53),
])

print("  Done.")

# === Arrange ===
print("Arranging...")
sections = [section_1, section_2, section_3, section_4, section_5, section_6]
full = arrange(sections, crossfade_seconds=0.0, sr=SR)

output_path = f"{OUTPUT}/cafe_day_theme_v4.wav"
export_arrangement(full, output_path, sr=SR, normalize=True, headroom_db=-1.0)

duration = len(full) / SR
print(f"\nExported: {output_path}")
print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
print(f"Structure: A1(sparse)→A2(build)→B1(full)→A3(breathe)→B2(full)→A4(resolve)")

# === Analyze ===
print("\nRunning analysis...")
import os
os.makedirs(f"{OUTPUT}/cafe_v4_analysis", exist_ok=True)
from src.analyze import full_analysis, detect_key
full_analysis(output_path, output_dir=f"{OUTPUT}/cafe_v4_analysis")
key = detect_key(output_path)
print(f"Detected key: {key['full']} ({key['confidence']:.3f})")
print("Done.")
