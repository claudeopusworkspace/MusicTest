"""
Café Day Theme v4 — Synth melody edition.

Replaces the independently-generated Foundation-1 melodies with a
programmatic synth lead that has coherent A and B melodies derived
from the same motivic material.

Keeps Foundation-1 layers for: guitar (now accompaniment), bass, pad.
Keeps sequenced drums from sample packs.
Adds: synth melody with related A/B phrases.
"""

import sys
sys.path.insert(0, "/workspace/MusicTest")

import numpy as np
import soundfile as sf
from pathlib import Path
from src.synth import Synth, render_melody_to_bars
from src.mixer import Mixer
from src.arrange import arrange, export_arrangement
from src.drums import load_sample, build_drum_track, beats_to_samples

# === Constants ===
SR = 44100
BPM = 100  # Foundation-1 layers were generated at 100 BPM (19.2s = 8 bars)
BARS = 8
BEATS_PER_BAR = 4
TOTAL_BEATS = BARS * BEATS_PER_BAR  # 32
SECTION_SECONDS = (60.0 / BPM) * TOTAL_BEATS  # 14.77s

GEN_DIR = "/workspace/MusicTest/generations"
SAMPLES_DIR = "/workspace/MusicTest/samples/cafe_percussion"
OUTPUT_DIR = "/workspace/MusicTest/output"

# === Synth Voice ===
lead_synth = Synth(
    waveform="warm",
    attack=0.04,
    decay=0.15,
    sustain=0.6,
    release=0.35,
    vibrato_rate=4.0,
    vibrato_depth=0.15,
    brightness=0.35,
    detune_cents=4.0,
)

# === Melody Composition ===
# Key: G major (G A B C D E F#)
# The A and B melodies share a core motif but develop it differently.
#
# Core motif: G4 → B4 → A4 (a gentle rising third then step down)
# A develops it into a relaxed, descending resolution
# B develops it into an ascending, more energetic continuation

# A melody: warm, relaxed, café feel (8 bars)
# Phrases breathe — rests between them, stepwise resolution
melody_a = [
    # Phrase 1 (bars 1-2): core motif → gentle descent
    ("G4",  1.0, 0.75), ("B4",  0.5, 0.7),  ("A4",  0.5, 0.65),
    ("B4",  1.0, 0.8),  ("G4",  0.5, 0.65), ("R",   0.5, 0),
    ("A4",  0.75, 0.7), ("G4",  0.75, 0.65), ("E4",  0.5, 0.6),
    ("R",   1.0, 0),

    # Phrase 2 (bars 3-4): answer — rising, then settles
    ("D4",  0.5, 0.65), ("E4",  0.5, 0.7),  ("G4",  1.0, 0.75),
    ("A4",  0.5, 0.7),  ("B4",  1.0, 0.8),  ("A4",  0.5, 0.65),
    ("G4",  1.5, 0.75), ("R",   0.5, 0),
    ("R",   1.0, 0),

    # Phrase 3 (bars 5-6): motif restated, slight variation
    ("G4",  0.5, 0.7),  ("B4",  1.0, 0.8),  ("A4",  0.5, 0.7),
    ("D5",  1.0, 0.85), ("B4",  0.5, 0.7),  ("A4",  0.5, 0.65),
    ("G4",  1.0, 0.75), ("R",   1.0, 0),
    ("R",   1.0, 0),

    # Phrase 4 (bars 7-8): resolution, long tones
    ("E4",  0.5, 0.6),  ("G4",  0.5, 0.65), ("A4",  1.0, 0.7),
    ("G4",  2.0, 0.75),
    ("R",   2.0, 0),
]

# B melody: more energy, higher register, same motif DNA
# Uses the G→B→A motif but pushes it upward and adds rhythmic drive
melody_b = [
    # Phrase 1 (bars 1-2): motif in higher octave, more momentum
    ("B4",  0.5, 0.8),  ("D5",  0.5, 0.85), ("E5",  1.0, 0.9),
    ("D5",  0.5, 0.8),  ("B4",  0.5, 0.75),
    ("D5",  0.5, 0.8),  ("E5",  0.5, 0.85), ("G5",  1.0, 0.9),
    ("E5",  0.5, 0.8),  ("D5",  0.5, 0.75),
    ("R",   1.0, 0),

    # Phrase 2 (bars 3-4): cascading descent (inversion of A's rise)
    ("G5",  0.5, 0.85), ("E5",  0.5, 0.8),  ("D5",  0.5, 0.8),
    ("B4",  1.0, 0.8),  ("A4",  0.5, 0.7),
    ("B4",  0.5, 0.75), ("D5",  1.0, 0.8),  ("B4",  0.5, 0.7),
    ("A4",  1.0, 0.7),
    ("R",   1.0, 0),

    # Phrase 3 (bars 5-6): call-and-response with itself
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

# Pad to exactly 32 beats each
def pad_to_beats(melody, target=32):
    total = sum(d for _, d, _ in melody)
    remaining = target - total
    if remaining > 0.01:
        melody.append(("R", remaining, 0))
    return melody

melody_a = pad_to_beats(melody_a)
melody_b = pad_to_beats(melody_b)

# Verify beat counts
beats_a = sum(d for _, d, _ in melody_a)
beats_b = sum(d for _, d, _ in melody_b)
print(f"Melody A: {beats_a} beats")
print(f"Melody B: {beats_b} beats")

# === Render Melodies ===
print("Rendering synth melodies...")
audio_melody_a = render_melody_to_bars(melody_a, lead_synth, bpm=BPM, bars=BARS, sr=SR)
audio_melody_b = render_melody_to_bars(melody_b, lead_synth, bpm=BPM, bars=BARS, sr=SR)

# Save standalone melody WAVs for analysis
Path(f"{OUTPUT_DIR}/cafe_v4_parts").mkdir(parents=True, exist_ok=True)
sf.write(f"{OUTPUT_DIR}/cafe_v4_parts/melody_a.wav", audio_melody_a, SR)
sf.write(f"{OUTPUT_DIR}/cafe_v4_parts/melody_b.wav", audio_melody_b, SR)
print("  Melodies rendered.")

# === Drum Patterns ===
# 4 variations for different section textures
print("Building drum patterns...")
kit = {
    "kick": f"{SAMPLES_DIR}/soft_kick.wav",
    "snare": f"{SAMPLES_DIR}/brush_snare.wav",
    "hat": f"{SAMPLES_DIR}/soft_hihat.wav",
}

# 16th note grid per bar, 8 bars total = 128 steps
def make_drum_pattern(kick_pat, snare_pat, hat_pat):
    """Build drum track from 1-bar patterns repeated 8 times."""
    return build_drum_track(
        kit=kit,
        patterns={
            "kick": kick_pat * BARS,
            "snare": snare_pat * BARS,
            "hat": hat_pat * BARS,
        },
        bpm=BPM, total_bars=BARS, sr=SR
    )

# All patterns: 16 chars = 16th-note grid for one bar
drums_sparse = make_drum_pattern(
    "x...........x...",
    "................",
    "..-..-..-..-.-.-",
)

drums_building = make_drum_pattern(
    "x.......x..x....",
    "....x.......x...",
    "x.-.x.-.x.-.x.-.",
)

drums_full = make_drum_pattern(
    "x..x....x..x.x.",
    "....x.......x.o.",
    "x.x.x.x.x.x.x.",
)

drums_breathing = make_drum_pattern(
    "x...............",
    "................",
    "...-........-.-.",
)

print("  Drums built.")

# === Mix Sections ===
# Foundation-1 layers (accompaniment — reduced volume since synth is lead)
print("Mixing sections...")

def mix_section(name, melody_audio, drums_audio,
                use_guitar=True, use_piano=False, use_pad=False,
                guitar_vol=0.5, piano_vol=0.4, pad_vol=0.35,
                bass_vol=0.6, melody_vol=0.75, drums_vol=0.45):
    """Mix a single section with the given layers."""
    mixer = Mixer(sr=SR)

    # Always include bass (foundation)
    mixer.add_track("bass", f"{GEN_DIR}/cafe_day_bass.wav", volume=bass_vol, pan=0.0)

    # Synth melody (lead)
    mixer.add_track("melody", melody_audio, volume=melody_vol, pan=0.0)

    # Drums
    mixer.add_track("drums", drums_audio, volume=drums_vol, pan=0.0)

    # Optional layers
    if use_guitar:
        mixer.add_track("guitar", f"{GEN_DIR}/cafe_day_guitar.wav",
                        volume=guitar_vol, pan=-0.2)
    if use_piano:
        mixer.add_track("piano", f"{GEN_DIR}/cafe_day_piano.wav",
                        volume=piano_vol, pan=0.25)
    if use_pad:
        mixer.add_track("pad", f"{GEN_DIR}/cafe_day_pad.wav",
                        volume=pad_vol, pan=0.0)

    mixed = mixer.mix(normalize=True, headroom_db=-1.0)
    return mixed

# Section structure:
# A1 (sparse)   — melody A + guitar + bass + sparse drums
# A2 (building) — melody A + guitar + piano + pad + bass + building drums
# B1 (full)     — melody B + guitar + piano + pad + bass + full drums
# A3 (breathe)  — melody A + bass + breathing drums (intimate moment)
# B2 (full)     — melody B + guitar + piano + pad + bass + full drums
# A4 (resolve)  — melody A + guitar + piano + pad + bass + building drums

section_a1 = mix_section("A1_sparse", audio_melody_a, drums_sparse,
                          use_guitar=True, use_piano=False, use_pad=False,
                          guitar_vol=0.45, melody_vol=0.7, drums_vol=0.35)

section_a2 = mix_section("A2_building", audio_melody_a, drums_building,
                          use_guitar=True, use_piano=True, use_pad=True,
                          guitar_vol=0.4, piano_vol=0.35, pad_vol=0.3,
                          melody_vol=0.7, drums_vol=0.4)

section_b1 = mix_section("B1_full", audio_melody_b, drums_full,
                          use_guitar=True, use_piano=True, use_pad=True,
                          guitar_vol=0.4, piano_vol=0.35, pad_vol=0.3,
                          melody_vol=0.75, drums_vol=0.45)

section_a3 = mix_section("A3_breathe", audio_melody_a, drums_breathing,
                          use_guitar=False, use_piano=False, use_pad=False,
                          bass_vol=0.5, melody_vol=0.65, drums_vol=0.3)

section_b2 = mix_section("B2_full", audio_melody_b, drums_full,
                          use_guitar=True, use_piano=True, use_pad=True,
                          guitar_vol=0.4, piano_vol=0.35, pad_vol=0.3,
                          melody_vol=0.75, drums_vol=0.45)

section_a4 = mix_section("A4_resolve", audio_melody_a, drums_building,
                          use_guitar=True, use_piano=True, use_pad=True,
                          guitar_vol=0.45, piano_vol=0.35, pad_vol=0.35,
                          melody_vol=0.7, drums_vol=0.4)

print("  All sections mixed.")

# === Arrange ===
print("Arranging final piece...")
sections = [section_a1, section_a2, section_b1, section_a3, section_b2, section_a4]
full = arrange(sections, crossfade_seconds=0.0, sr=SR)

output_path = f"{OUTPUT_DIR}/cafe_day_theme_v4.wav"
export_arrangement(full, output_path, sr=SR, normalize=True, headroom_db=-1.0)

duration = len(full) / SR
print(f"\nExported: {output_path}")
print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
print(f"Sections: A1→A2→B1→A3→B2→A4 (6 × {SECTION_SECONDS:.1f}s)")

# === Analyze ===
print("\nRunning analysis...")
from src.analyze import full_analysis
full_analysis(output_path, output_dir=f"{OUTPUT_DIR}/cafe_v4_analysis")
print("Analysis complete.")
