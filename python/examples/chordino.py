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

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List

from vamprust.audio import AudioProcessor, load_audio


def print_chords(features: List[dict[str, Any]]) -> None:
    for feature in features:
        timestamp = feature["sec"] + feature["nsec"] / 1e9
        chord = feature["label"]
        print(f"{timestamp:.2f}s: {chord}")


def main() -> None:
    parser = ArgumentParser(description="Extract chords using Chordino Vamp plugin")
    parser.add_argument(
        "audio_file", type=Path, help="Path to input audio file (e.g., WAV, MP3)"
    )
    args = parser.parse_args()
    if not args.audio_file.exists():
        print(f"Audio file '{args.audio_file}' does not exist")
        return

    AUDIO_PATH = Path(args.audio_file)
    audio, sr = load_audio(AUDIO_PATH, sample_rate=44100)
    print(
        f"Loaded audio from {AUDIO_PATH}, sample rate: {sr}, length: {len(audio)} samples"
    )

    processor = AudioProcessor()
    print("Created AudioProcessor")
    features = processor.process_audio(
        "chordino",
        audio,
        sample_rate=sr,
        output_index=0,
        parameters={
            "useNNLS": 1,
            "rollon": 1,
            "tuningmode": 0.0,
            "whitening": 1,
            "s": 0.7,
            "boostn": 0.1,
        },
    )

    print(f"Got {len(features)} features")
    if features:
        print_chords(features)

if __name__ == "__main__":
    main()
