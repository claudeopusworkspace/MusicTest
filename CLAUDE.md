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

## Conventions
- All audio work uses 44100 Hz sample rate (Foundation-1's native rate), WAV format unless otherwise specified
- Tempo/key metadata should be preserved in filenames or sidecar JSON
