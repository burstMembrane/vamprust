#!/usr/bin/env python3
"""
Audio file processing example for VampRust Python bindings.

This example demonstrates:
1. Loading audio files
2. High-level audio processing with AudioProcessor
3. Using FeatureSet for analysis
4. Saving results to different formats

Requirements: pip install soundfile
Optional: pip install librosa pandas
"""

import sys
from pathlib import Path

import numpy as np
from vamprust import AudioProcessor, FeatureSet, load_audio, process_audio_file


def generate_test_audio(
    filename: str = "test_audio.wav", duration: float = 5.0, sample_rate: int = 44100
) -> str:
    """Generate a test audio file with multiple frequency components."""

    import soundfile as sf

    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Create a complex signal with multiple frequencies
    signal = (
        0.3 * np.sin(2 * np.pi * 440 * t)  # A4
        + 0.2 * np.sin(2 * np.pi * 880 * t)  # A5
        + 0.1 * np.sin(2 * np.pi * 1320 * t)  # E6
        + 0.05 * np.random.randn(len(t))  # Small amount of noise
    )

    # Add some amplitude modulation
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))  # 2 Hz modulation
    signal *= envelope

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    sf.write(filename, signal, sample_rate)
    return filename


def main() -> None:
    print("VampRust Python Bindings - Audio File Processing Example")
    print("=" * 60)

    # Generate test audio if no file provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = generate_test_audio()
        if not audio_file:
            return

    if not Path(audio_file).exists():
        print(f"Audio file not found: {audio_file}")
        return

    print(f"Processing audio file: {audio_file}")

    # Create audio processor
    processor = AudioProcessor()

    # Discover available plugins
    print("\nDiscovering plugins...")
    plugins = processor.discover_plugins()

    if not plugins:
        print("No plugins found. Please install some Vamp plugins.")
        return

    print(f"Found {len(plugins)} plugins:")
    for i, plugin in enumerate(plugins[:5]):  # Show first 5
        print(f"  {i + 1}: {plugin.identifier} - {plugin.name}")

    # Try to find a spectral analysis plugin
    spectral_plugins = processor.get_plugins_by_type("spectral")
    onset_plugins = processor.get_plugins_by_type("onset")
    tempo_plugins = processor.get_plugins_by_type("tempo")
    print(f"\nSpectral plugins found: {len(spectral_plugins)}")
    print(f"Onset plugins found: {len(onset_plugins)}")
    print(f"Tempo plugins found: {len(tempo_plugins)}")
    # Choose a plugin to demonstrate
    demo_plugin = None
    if spectral_plugins:
        demo_plugin = spectral_plugins[0]
        print(f"\nUsing spectral analysis plugin: {demo_plugin.identifier}")
    elif onset_plugins:
        demo_plugin = onset_plugins[0]
        print(f"\nUsing onset detection plugin: {demo_plugin.identifier}")
    elif tempo_plugins:
        demo_plugin = tempo_plugins[0]
        print(f"\nUsing tempo analysis plugin: {demo_plugin.identifier}")
    else:
        demo_plugin = plugins[0]
        print(f"\nUsing first available plugin: {demo_plugin.identifier}")

    # Method 1: High-level processing
    print("\n--- Method 1: High-level processing ---")
    try:
        features_raw = process_audio_file(audio_file, demo_plugin.identifier)
        feature_set = FeatureSet(features_raw)

        print(f"Extracted {len(feature_set)} features")
        print(f"Feature set summary: {feature_set}")

        # Show summary statistics
        summary = feature_set.summary()
        print("\nSummary:")
        for key, value in summary.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} items")
            else:
                print(f"  {key}: {value}")

        # Export to different formats
        if len(feature_set) > 0:
            print("\nFirst few features:")
            for i, feature in enumerate(feature_set[:3]):
                print(f"  Feature {i + 1}: {len(feature.values)} values")
                if feature.has_timestamp:
                    print(f"    Timestamp: {feature.timestamp:.3f}s")
                if feature.values:
                    print(f"    Values: {feature.values[:3]}...")

        # Try to export to pandas if available
        try:
            df = feature_set.to_pandas()
            print(f"\nPandas DataFrame shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
        except ImportError:
            print("\nPandas not available for DataFrame export")

    except Exception as e:
        print(f"High-level processing failed: {e}")

    # Method 2: Manual processing with AudioProcessor
    print("\n--- Method 2: Manual processing ---")
    try:
        # Load audio manually
        audio, sample_rate = load_audio(audio_file)
        print(f"Loaded audio: {audio.shape} samples at {sample_rate} Hz")

        # Process with specific parameters
        features_raw = processor.process_audio(
            demo_plugin.identifier, audio, sample_rate, block_size=1024, step_size=512
        )

        print(f"Manual processing extracted {len(features_raw)} features")

        # Group features by label if they have labels
        feature_set = FeatureSet(features_raw)
        if any(f.label for f in feature_set):
            groups = feature_set.group_by_label()
            print(f"Feature groups: {list(groups.keys())}")
            for label, group in groups.items():
                print(f"  {label}: {len(group)} features")

    except Exception as e:
        print(f"Manual processing failed: {e}")

    # Method 3: Low-level processing
    print("\n--- Method 3: Low-level processing ---")
    try:
        # Load plugin manually
        plugin = processor.load_plugin(demo_plugin.identifier, sample_rate)
        if plugin:
            print(f"Loaded plugin: {plugin}")
            print(f"Input domain: {plugin.get_input_domain()}")

            # Initialize with custom parameters
            channels = 1 if audio.ndim == 1 else audio.shape[0]
            block_size = 2048
            step_size = 1024

            if plugin.initialise(channels, step_size, block_size):
                print(
                    f"Initialized with {channels} channels, block={block_size}, step={step_size}"
                )

                # Process a few blocks manually
                if audio.ndim == 1:
                    audio = audio.reshape(1, -1)

                for i, block_start in enumerate(
                    range(0, min(audio.shape[1], 5 * step_size), step_size)
                ):
                    block_end = min(block_start + block_size, audio.shape[1])
                    block = audio[:, block_start:block_end]

                    # Pad if necessary
                    if block.shape[1] < block_size:
                        padded = np.zeros((block.shape[0], block_size))
                        padded[:, : block.shape[1]] = block
                        block = padded

                    input_buffers = [channel.tolist() for channel in block]

                    sec = int(block_start / sample_rate)
                    nsec = int((block_start / sample_rate - sec) * 1e9)

                    features = plugin.process(input_buffers, sec, nsec)
                    if features:
                        print(f"  Block {i + 1}: {features['feature_count']} features")
                        if i >= 2:  # Just show first 3 blocks
                            break
            else:
                print("Failed to initialize plugin")
        else:
            print("Failed to load plugin")

    except Exception as e:
        print(f"Low-level processing failed: {e}")

    print("\nProcessing completed!")


if __name__ == "__main__":
    main()
