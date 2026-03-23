"""
Café Day Theme v4 — Improved B-section guitar via pitch-profile search.

Changes from v3:
- guitar_b replaced with seed 17 (corr=0.949 with guitar_a, vs old 0.563)
  This gives the B sections a guitar that shares guitar_a's melodic DNA
  (D-heavy, G major diatonic) while still being a distinct musical take.

Everything else matches v3 exactly — same volumes, same layer assignments,
same section structure. This isolates the single variable being tested:
does a more melodically-related guitar_b improve the composition?

Key: G major | BPM: 100 | 8 bars per section (19.2s) | 6 sections
Structure: A1(sparse) → A2(build) → B1(full) → A3(breathe) → B2(full) → A4(resolve)

Layer sources:
  guitar_a:  generations/cafe_day_guitar.wav (original, key=D major, 30 notes)
  guitar_b:  generations/cafe_day_guitar_b_v4.wav (seed 17, key=D major, 57 notes, corr=0.949)
  bass:      generations/cafe_day_bass.wav
  piano_a:   generations/cafe_day_piano.wav (seed s77)
  piano_b:   generations/cafe_day_piano_b.wav (seed s22)
  pad:       generations/cafe_day_pad.wav
  drums:     generations/cafe_drums_{sparse,building,full,breathing}.wav

Volume levels from v3 least-squares reconstruction (cafe_v3_reconstruct.py).
"""

import sys
sys.path.insert(0, "/workspace/MusicTest")

import os
from src.mixer import Mixer
from src.arrange import arrange, export_arrangement

SR = 44100
BPM = 100
BARS = 8

GEN = "generations"
OUTPUT = "output"

# === Layer paths ===
GUITAR_A = f"{GEN}/cafe_day_guitar.wav"
GUITAR_B = f"{GEN}/cafe_day_guitar_b_v4.wav"   # NEW: seed 17 (corr=0.949)
BASS     = f"{GEN}/cafe_day_bass.wav"
PIANO_A  = f"{GEN}/cafe_day_piano.wav"
PIANO_B  = f"{GEN}/cafe_day_piano_b.wav"
PAD      = f"{GEN}/cafe_day_pad.wav"
DRUMS_SPARSE    = f"{GEN}/cafe_drums_sparse.wav"
DRUMS_BUILDING  = f"{GEN}/cafe_drums_building.wav"
DRUMS_FULL      = f"{GEN}/cafe_drums_full.wav"
DRUMS_BREATHING = f"{GEN}/cafe_drums_breathing.wav"


def mix_section(layers_config):
    """Mix a section. layers_config: list of (name, path, volume) tuples."""
    mixer = Mixer(sr=SR)
    for name, path, volume in layers_config:
        mixer.add_track(name, path, volume=volume, pan=0.0)
    return mixer.mix(normalize=True, headroom_db=-1.0)


# === Section Definitions ===
# Volumes identical to v3 reconstruction

# S1: Sparse opening — guitar + bass + sparse drums
section_1 = mix_section([
    ("guitar",  GUITAR_A,       0.840),
    ("bass",    BASS,            0.473),
    ("drums",   DRUMS_SPARSE,    0.368),
])

# S2: Building — add piano + pad, fuller drums
section_2 = mix_section([
    ("guitar",  GUITAR_A,        0.840),
    ("bass",    BASS,             0.525),
    ("piano",   PIANO_A,          0.409),
    ("pad",     PAD,              0.158),
    ("drums",   DRUMS_BUILDING,   0.473),
])

# S3: B section full — NEW guitar_b + piano_b + pad + full drums
section_3 = mix_section([
    ("guitar",  GUITAR_B,        0.783),   # seed 17
    ("bass",    BASS,             0.579),
    ("piano",   PIANO_B,          0.578),
    ("pad",     PAD,              0.261),
    ("drums",   DRUMS_FULL,       0.525),
])

# S4: Breathing — back to guitar A, minimal drums
section_4 = mix_section([
    ("guitar",  GUITAR_A,         0.895),
    ("bass",    BASS,              0.526),
    ("drums",   DRUMS_BREATHING,   0.315),
])

# S5: B section full (repeat of S3, slight pad increase)
section_5 = mix_section([
    ("guitar",  GUITAR_B,        0.783),   # seed 17
    ("bass",    BASS,             0.579),
    ("piano",   PIANO_B,          0.578),
    ("pad",     PAD,              0.314),
    ("drums",   DRUMS_FULL,       0.525),
])

# S6: Resolution — guitar A returns with piano + pad + building drums
section_6 = mix_section([
    ("guitar",  GUITAR_A,         0.840),
    ("bass",    BASS,              0.578),
    ("piano",   PIANO_A,           0.460),
    ("pad",     PAD,               0.263),
    ("drums",   DRUMS_BUILDING,    0.525),
])

# === Arrange ===
sections = [section_1, section_2, section_3, section_4, section_5, section_6]
full = arrange(sections, crossfade_seconds=0.0, sr=SR)

output_path = f"{OUTPUT}/cafe_day_theme_v4.wav"
export_arrangement(full, output_path, sr=SR, normalize=True, headroom_db=-1.0)

duration = len(full) / SR
print(f"Exported: {output_path}")
print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
print(f"Structure: A1(sparse) → A2(build) → B1(full) → A3(breathe) → B2(full) → A4(resolve)")
print(f"Change: guitar_b replaced with seed 17 (corr=0.949 vs old 0.563)")

# === Analyze ===
os.makedirs(f"{OUTPUT}/cafe_v4_analysis", exist_ok=True)
from src.analyze import full_analysis, detect_key
full_analysis(output_path, output_dir=f"{OUTPUT}/cafe_v4_analysis")
key = detect_key(output_path)
print(f"Detected key: {key['full']} ({key['confidence']:.3f})")
