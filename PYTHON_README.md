# VampRust Python Bindings

Python bindings for VampRust, providing a high-level interface to the Vamp Plugin SDK for audio analysis.

## Installation

### Requirements

- Python 3.8+
- Rust (for building from source)
- CMake (for building the Vamp SDK)
- Vamp plugins installed on your system

### From Source

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/yourusername/vamprust.git
cd vamprust

# Install Python dependencies
pip install maturin

# Build and install the Python package
maturin develop --features python

# Or for release builds
maturin build --release --features python
pip install target/wheels/vamprust-*.whl
```

### Optional Dependencies

```bash
# For audio file loading
pip install soundfile

# For resampling support
pip install librosa

# For DataFrame export
pip install pandas

# For development and testing
pip install pytest pytest-benchmark
```

## Quick Start

```python
import vamprust
import numpy as np

# Create a host and discover plugins
host = vamprust.VampHost()
libraries = host.find_plugin_libraries()

if libraries:
    # Load the first library
    library = host.load_library(libraries[0])
    plugins = library.list_plugins()
    
    if plugins:
        # Instantiate a plugin
        plugin = library.instantiate_plugin(plugins[0].index, 44100.0)
        
        if plugin and plugin.initialise(1, 512, 1024):
            # Generate test audio (440Hz sine wave)
            samples = 1024
            audio = 0.5 * np.sin(2 * np.pi * 440 * np.arange(samples) / 44100.0)
            
            # Process audio
            features = plugin.process([audio.tolist()], 0, 0)
            print(f"Extracted {features['feature_count']} features")
```

## High-Level API

The high-level API provides convenient functions for common audio analysis tasks:

```python
from vamprust import AudioProcessor, process_audio_file, FeatureSet

# Process an audio file directly
features_raw = process_audio_file("audio.wav", "plugin:identifier")
feature_set = FeatureSet(features_raw)

# Or use AudioProcessor for more control
processor = AudioProcessor()
plugins = processor.discover_plugins()

# Load audio and process
audio, sample_rate = vamprust.load_audio("audio.wav")
features_raw = processor.process_audio("plugin:identifier", audio, sample_rate)
feature_set = FeatureSet(features_raw)

# Analyze features
print(f"Extracted {len(feature_set)} features")
print(f"Summary: {feature_set.summary()}")

# Export to different formats
numpy_array = feature_set.to_numpy()
dict_list = feature_set.to_dict_list()
dataframe = feature_set.to_pandas()  # requires pandas
```

## Feature Analysis

The `FeatureSet` class provides powerful tools for analyzing extracted features:

```python
# Filter features by timestamp
early_features = feature_set.filter_by_timestamp(0.0, 5.0)  # First 5 seconds

# Filter by label
onset_features = feature_set.filter_by_label("onset")

# Group by label
groups = feature_set.group_by_label()
for label, group in groups.items():
    print(f"{label}: {len(group)} features")

# Get summary statistics
summary = feature_set.summary()
print(f"Duration: {summary['duration']:.2f} seconds")
print(f"Feature dimensions: {summary['dimensions']}")
print(f"Mean values: {summary['mean_values']}")
```

## Plugin Discovery

VampRust automatically searches for plugins in standard locations:

- **macOS**: `/Library/Audio/Plug-Ins/Vamp`, `~/Library/Audio/Plug-Ins/Vamp`
- **Linux**: `/usr/lib/vamp`, `/usr/local/lib/vamp`, `~/.vamp`

You can also set the `VAMP_PATH` environment variable to specify custom search paths.

```python
# Discover all available plugins
processor = AudioProcessor()
plugins = processor.discover_plugins()

for plugin in plugins:
    print(f"{plugin.identifier}: {plugin.name}")
    print(f"  Library: {plugin.library_path}")
```

## Examples

### Basic Plugin Usage

```python
import vamprust
import numpy as np

# Create host and find plugins
host = vamprust.VampHost()
processor = vamprust.AudioProcessor()
plugins = processor.discover_plugins()

# Find a spectral analysis plugin
spectral_plugin = next((p for p in plugins if 'spectral' in p.identifier.lower()), None)

if spectral_plugin:
    # Generate test signal
    duration = 2.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    
    # Process signal
    features = processor.process_audio(spectral_plugin.identifier, signal, sample_rate)
    feature_set = vamprust.FeatureSet(features)
    
    print(f"Processed {duration}s signal, extracted {len(feature_set)} features")
```

### Audio File Processing

```python
import vamprust

# Process an audio file for onset detection
try:
    features = vamprust.process_audio_file("song.wav", "vamp:vamp-example-plugins:onsetdetector")
    feature_set = vamprust.FeatureSet(features)
    
    # Filter onset events
    onsets = feature_set.filter_by_label("onset")
    onset_times = [f.timestamp for f in onsets if f.has_timestamp]
    
    print(f"Found {len(onset_times)} onsets at times: {onset_times[:5]}...")
    
except Exception as e:
    print(f"Processing failed: {e}")
```

### Batch Processing

```python
import vamprust
from pathlib import Path

def analyze_directory(audio_dir, plugin_id, output_file=None):
    """Analyze all audio files in a directory."""
    processor = vamprust.AudioProcessor()
    results = {}
    
    for audio_file in Path(audio_dir).glob("*.wav"):
        try:
            features = vamprust.process_audio_file(str(audio_file), plugin_id)
            feature_set = vamprust.FeatureSet(features)
            results[audio_file.name] = feature_set.summary()
            print(f"Processed {audio_file.name}: {len(feature_set)} features")
        except Exception as e:
            print(f"Failed to process {audio_file.name}: {e}")
    
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

# Usage
results = analyze_directory("audio_files/", "vamp:plugin:identifier", "analysis_results.json")
```

## Low-Level API

For advanced users, the low-level API provides direct access to Vamp plugin functionality:

```python
import vamprust

# Manual plugin management
host = vamprust.VampHost()
library = host.load_library("/path/to/plugin.so")

if library:
    plugins = library.list_plugins()
    plugin = library.instantiate_plugin(0, 44100.0)
    
    if plugin:
        print(f"Input domain: {plugin.get_input_domain()}")
        print(f"Preferred block size: {plugin.get_preferred_block_size()}")
        
        # Initialize with specific parameters
        if plugin.initialise(channels=2, step_size=512, block_size=1024):
            # Process audio blocks manually
            for block_data in audio_blocks:
                features = plugin.process(block_data, sec, nsec)
                # Process features...
            
            # Get remaining features
            remaining = plugin.get_remaining_features()
```

## Error Handling

VampRust provides comprehensive error handling:

```python
try:
    features = vamprust.process_audio_file("nonexistent.wav", "invalid:plugin")
except vamprust.VampError as e:
    print(f"Vamp error: {e}")
except FileNotFoundError:
    print("Audio file not found")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Tips

1. **Reuse AudioProcessor instances** - Plugin discovery is expensive
2. **Use appropriate block sizes** - Larger blocks = more efficient processing
3. **Process in batches** - Group similar processing tasks together
4. **Choose the right step size** - Balance between time resolution and performance

```python
# Good: Reuse processor
processor = vamprust.AudioProcessor()
processor.discover_plugins()  # Do this once

for audio_file in audio_files:
    features = processor.process_audio(plugin_id, audio, sample_rate)
    # Process features...

# Good: Efficient block processing
features = processor.process_audio(
    plugin_id, 
    audio, 
    sample_rate,
    block_size=2048,    # Larger blocks for efficiency
    step_size=1024      # 50% overlap for good time resolution
)
```

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest python/tests/

# Run with coverage
pip install pytest-cov
pytest --cov=vamprust python/tests/
```

## Troubleshooting

### Common Issues

1. **No plugins found**
   - Install Vamp plugins: `brew install vamp-plugin-sdk` (macOS) or `apt-get install vamp-plugin-sdk` (Linux)
   - Check `VAMP_PATH` environment variable
   - Verify plugin installation with `vamp-simple-host -l`

2. **Plugin loading fails**
   - Check plugin compatibility (32-bit vs 64-bit)
   - Verify plugin dependencies are installed
   - Try loading plugins manually with the low-level API

3. **Audio loading fails**
   - Install `soundfile`: `pip install soundfile`
   - For resampling support: `pip install librosa`
   - Check audio file format compatibility

4. **Build failures**
   - Ensure CMake is installed
   - Check that git submodules are initialized: `git submodule update --init --recursive`
   - Verify Rust toolchain is up to date: `rustup update`

### Getting Help

- Check the main VampRust README for Vamp SDK installation
- Run examples to verify installation: `python python/examples/basic_usage.py`
- For plugin-specific issues, consult the plugin documentation

## License

This project is licensed under the same terms as the Vamp Plugin SDK (BSD-3-Clause).