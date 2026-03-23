"""
Guitar B candidate search — find a B-section guitar that melodically
relates to guitar_a by generating many seeds and scoring pitch profiles.

Target: pitch class correlation > 0.7 with guitar_a while still sounding
like a distinct take (not identical).
"""

import sys
sys.path.insert(0, "/workspace/MusicTest")

import numpy as np
import json
from pathlib import Path
from src.generate import generate, save_wav, load_model, build_prompt

# Suppress basic_pitch warnings
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)
from basic_pitch.inference import predict

# === Reference profile: guitar_a ===
print("Extracting guitar_a reference profile...")
_, _, notes_a = predict("generations/cafe_day_guitar.wav")
pitches_a = [n[2] for n in notes_a]
durs_a = [n[1] - n[0] for n in notes_a]
pc_a = np.zeros(12)
for p, d in zip(pitches_a, durs_a):
    pc_a[p % 12] += d
pc_a /= pc_a.sum()
range_a = (min(pitches_a), max(pitches_a))
density_a = len(notes_a)

print(f"  Guitar A: {density_a} notes, range MIDI {range_a[0]}-{range_a[1]}")
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
top_a = sorted(range(12), key=lambda i: pc_a[i], reverse=True)[:3]
print(f"  Top pitches: {', '.join(f'{NOTE_NAMES[i]}={pc_a[i]:.1%}' for i in top_a)}")

# === Current guitar_b baseline ===
_, _, notes_b_old = predict("generations/cafe_day_guitar_b.wav")
pc_b_old = np.zeros(12)
for n in notes_b_old:
    pc_b_old[n[2] % 12] += n[1] - n[0]
pc_b_old /= pc_b_old.sum()
baseline_corr = np.corrcoef(pc_a, pc_b_old)[0, 1]
print(f"  Current guitar_b correlation: {baseline_corr:.3f}")
print()

# === Generate candidates ===
print("Loading Foundation-1...")
load_model()
print("Ready.\n")

OUTPUT_DIR = Path("experiments/guitar_b_candidates")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prompt matching the café guitar style
PROMPT = build_prompt(
    instrument="Acoustic guitar",
    timbre="warm clean tone",
    behavior="gentle fingerpicking",
    key="G major",
    bars=8,
    bpm=100,
)
print(f"Prompt: {PROMPT}")
print()

BATCH_SIZE = 20
results = []
best_corr = baseline_corr
best_seed = None


def score_candidate(wav_path):
    """Extract pitch profile and score against guitar_a."""
    try:
        _, _, notes = predict(str(wav_path))
        if len(notes) < 5:
            return None  # Too sparse to compare

        pitches = [n[2] for n in notes]
        durs = [n[1] - n[0] for n in notes]
        pc = np.zeros(12)
        for p, d in zip(pitches, durs):
            pc[p % 12] += d
        if pc.sum() == 0:
            return None
        pc /= pc.sum()

        # Metrics
        corr = float(np.corrcoef(pc_a, pc)[0, 1])
        pitch_range = (min(pitches), max(pitches))
        range_overlap = (max(range_a[0], pitch_range[0]),
                         min(range_a[1], pitch_range[1]))
        range_score = max(0, range_overlap[1] - range_overlap[0]) / (range_a[1] - range_a[0])
        density_ratio = len(notes) / density_a  # 1.0 = same density

        return {
            "corr": corr,
            "notes": len(notes),
            "range": pitch_range,
            "range_score": range_score,
            "density_ratio": density_ratio,
            "pc": pc.tolist(),
        }
    except Exception as e:
        print(f"    Score error: {e}")
        return None


# Generate in batches of 20, up to 60 max
for batch in range(3):
    batch_start = batch * BATCH_SIZE
    print(f"=== Batch {batch + 1} (seeds {batch_start}-{batch_start + BATCH_SIZE - 1}) ===")

    for i in range(BATCH_SIZE):
        seed = batch_start + i
        wav_path = OUTPUT_DIR / f"guitar_b_s{seed}.wav"

        # Generate
        audio, sr = generate(PROMPT, bars=8, bpm=100, seed=seed)
        save_wav(audio, sr, str(wav_path))

        # Score
        score = score_candidate(wav_path)
        if score is None:
            print(f"  seed={seed:3d}: [skipped - too sparse]")
            continue

        score["seed"] = seed
        results.append(score)

        marker = ""
        if score["corr"] > best_corr:
            best_corr = score["corr"]
            best_seed = seed
            marker = " *** NEW BEST ***"

        print(f"  seed={seed:3d}: corr={score['corr']:.3f}  notes={score['notes']:3d}  "
              f"density={score['density_ratio']:.1f}x  range={score['range']}{marker}")

    # Report batch results
    print(f"\n  Best so far: seed={best_seed}, corr={best_corr:.3f}")
    print(f"  (baseline guitar_b was {baseline_corr:.3f})")

    # Early stop if we found something really good
    if best_corr > 0.85:
        print(f"\n  Excellent match found (>{0.85:.0%}), stopping early.")
        break

    print()

# === Final ranking ===
results.sort(key=lambda r: r["corr"], reverse=True)
print("\n" + "=" * 60)
print("TOP 5 CANDIDATES:")
print("=" * 60)
for i, r in enumerate(results[:5]):
    print(f"  #{i+1}: seed={r['seed']:3d}  corr={r['corr']:.3f}  "
          f"notes={r['notes']:3d}  density={r['density_ratio']:.1f}x  "
          f"range=MIDI {r['range'][0]}-{r['range'][1]}")
    pc = r["pc"]
    top = sorted(range(12), key=lambda j: pc[j], reverse=True)[:3]
    top_str = ", ".join(f"{NOTE_NAMES[j]}={pc[j]:.1%}" for j in top)
    print(f"       Top: {top_str}")

print(f"\nBaseline guitar_b: corr={baseline_corr:.3f}")
print(f"Best candidate: seed={best_seed}, corr={best_corr:.3f}")

# Save results
with open(str(OUTPUT_DIR / "search_results.json"), "w") as f:
    json.dump({"reference": "guitar_a", "baseline_corr": baseline_corr,
               "best_seed": best_seed, "best_corr": best_corr,
               "results": results}, f, indent=2)
print(f"\nResults saved to {OUTPUT_DIR}/search_results.json")
