# MusicTest

## What This Is
A toolkit for composing music using Foundation-1 (text-to-sample AI model) combined with programmatic mixing, drum sequencing, and audio analysis. The goal is to enable Claude to compose songs by generating individual instrument samples, sequencing percussion from WAV hits, mixing everything together, and analyzing results via spectrograms/chromagrams.

## Architecture
- **Python 3.10** virtual environment (required by stable-audio-tools dependency chain)
- **Foundation-1** model via RC-stable-audio-tools fork for sample generation
- Custom Python modules for mixing, sequencing, and analysis

## Key Directories
- `models/` — Foundation-1 model weights (gitignored, large files)
- `generations/` — raw generated samples (gitignored)
- `output/` — mixed/final audio (gitignored)
- `samples/` — drum/percussion sample packs (committed, small WAVs)
- `src/` — project source code

## Commands
- Activate venv: `source .venv/bin/activate`
- Run tests: `python -m pytest tests/`

## Analyzing Generated Audio
Claude cannot play audio, but can interpret visual analysis images. After generating
any sample, run `full_analysis()` from `src.analyze` to produce the standard suite
of 7 images, then read them to assess the output.

### Standard Analysis (always run)
| Image | What it tells you |
|---|---|
| Mel Spectrogram | Instrument timbre, harmonic structure, frequency distribution |
| Chromagram | Chord changes and harmonic movement over time |
| Waveform | Clipping, silence, amplitude envelope, gross dynamics |
| Pitch Histogram | **Key identification** — compare top notes against known triads |
| RMS Energy | Perceived loudness over time — reveals rising/falling dynamics |
| Spectral Centroid | Brightness over time — detects filter sweeps, timbral evolution |
| Tempogram | Precise BPM with visual confirmation of tempo stability |

### Diagnostic Tools (use when investigating a specific question)
- `zoomed_spectrogram()` — High time-resolution view of a short window around a
  transient. Use to check for reverb tails, attack character, artifacts, or FX.
  Auto-selects the loudest transient if no center_time is given. Not part of
  full_analysis because a single zoomed window may not represent the whole sample.

### Interpretation Workflow
1. Generate a sample with `src.generate`
2. Run `full_analysis()` to produce all 7 images
3. Read images to assess: correct key? right tempo? expected timbre/dynamics?
4. If a specific question remains, use the diagnostic tool for a closer look
5. Use `detect_key()` and `detect_tempo()` for numeric confirmation

## Composition Workflow

### 1. Generate Layers
Each instrument is generated independently via Foundation-1 (`src.generate`).
- Always specify key, bars, BPM in the prompt
- Supported BPMs: 100, 110, 120, 128, 130, 140, 150
- Generate 3-5 seeds per layer, analyze each, pick the best key/tempo match
- Foundation-1 does NOT generate percussion — sequence drums from WAV samples

### 2. Seed Audition
Run `detect_key()` and `detect_tempo()` on each candidate. Score by:
- Correct key (exact match > related key > wrong key)
- Correct tempo (within ~2 BPM of target)

### 3. Percussion
Build drum patterns with `src.drums`. Create variation:
- Sparse pattern (kick + light hat) for quiet sections
- Building pattern (add snare, fuller hats)
- Full pattern (ghost notes, busier hats)
- Breathing pattern (minimal, spacious)

### 4. Mix Sections
Use `src.mixer` to combine layers at different volumes/pans per section.
**Layer orchestration is critical for texture** — not every instrument should
play in every section. A typical arc:

| Section | Layers | Character |
|---|---|---|
| A1 (sparse) | Guitar + bass + light drums | Intimate opening |
| A2 (building) | + piano, pad, fuller drums | Warming up |
| B1 (full) | Melody guitar + piano + all | Peak energy |
| A3 (breathing) | Guitar + bass + minimal drums | Exhale moment |
| B2 (full) | Melody guitar + piano + all | Second peak |
| A4 (resolution) | Full but settled | Warm close |

This arc pattern (sparse → build → peak → breathe → peak → resolve) is a
good starting point for most pieces. Adjust per song — the principle is that
texture should breathe, not flatline.

### 5. Arrange
Use `src.arrange` to concatenate sections. Key rules:
- **Use hard cuts (crossfade_seconds=0.0)** for beat-aligned sections.
  Crossfades blend two harmonic contexts and sound worse than clean cuts.
- **Don't use make_loopable()** — if sections loop cleanly individually,
  concatenation at beat boundaries is already seamless.
- Shared layers (bass, pad, drums) across sections provide continuity
  that makes hard cuts between different melodic content feel natural.

### 6. Analyze Final Mix
Run `full_analysis()` on the exported arrangement. Check:
- RMS energy shows the intended dynamic arc (not flatlined)
- Pitch histogram confirms overall key
- Chromagram shows section contrast (A vs B harmonic emphasis)
- No clipping in waveform

## Conventions
- All audio work uses 44100 Hz sample rate (Foundation-1's native rate), WAV format unless otherwise specified
- Tempo/key metadata should be preserved in filenames or sidecar JSON
