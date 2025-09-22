# parameters from here http://www.isophonics.net/nnls-chroma
"""
Suggested Parameter Settings

generic pop song:
    use approximate transcription (NNLS): on
    spectral roll-on: 1.0%
    tuning mode: global tuning
    spectral whitening: 1.0
    spectral shape: 0.7

solo harpsichord:
    use approximate transcription (NNLS): on
    spectral roll-on: 1.0%
    tuning mode: global tuning
    spectral whitening: 0.4
    spectral shape: 0.9

generic pop song (quick and dirty):
    use approximate transcription (NNLS): off
    spectral roll-on: 1.0%
    tuning mode: global tuning
    spectral whitening: 1.0
    spectral shape: (doesn't matter: no NNLS)


Chord extractor parameters:
  defaults = {
    'useNNLS': 1,
    'rollon': 1,
    'tuningmode': "global",
    'whitening': 1,
    's': 0.7,
    # don't know what this one is
    'boostn': 0.1,
}

"""

import json
from pathlib import Path

from vamprust.audio import AudioProcessor, load_audio


def main() -> None:
    AUDIO_PATH = Path(__file__).parent.parent.parent / "mix.wav"
    audio, sr = load_audio(AUDIO_PATH, sample_rate=16000)

    features = AudioProcessor().process_audio(
        "rms",
        audio,
        sample_rate=sr,
        output_index=0,
    )
    print(f"Extracted {len(features)} features")
    # save to json
    print(features)


if __name__ == "__main__":
    main()
