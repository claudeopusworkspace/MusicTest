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

## Conventions
- All audio work uses 44100 Hz sample rate (Foundation-1's native rate), WAV format unless otherwise specified
- Tempo/key metadata should be preserved in filenames or sidecar JSON
