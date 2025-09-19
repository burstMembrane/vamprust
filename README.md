# VampRust

A Rust wrapper for the [Vamp Plugin SDK](https://github.com/vamp-plugins/vamp-plugin-sdk) that allows you to run C++ Vamp audio analysis plugins from Rust using bindgen.

## Features

- Safe Rust bindings for the Vamp Plugin SDK C API
- Plugin discovery and enumeration
- Audio processing with automatic memory management
- Cross-platform support (macOS, Linux)
- No C++ shim required - uses the SDK's native C ABI
- **Bundled SDK** - includes Vamp SDK as a submodule, no external dependencies required!

## Quick Start

This project includes the Vamp Plugin SDK as a git submodule, so you don't need to install anything manually:

```bash
# Clone with submodules
git clone --recursive https://github.com/yourusername/vamprust.git
cd vamprust

# Or if you already cloned without --recursive:
git submodule update --init --recursive

# Build and run example (requires cmake)
cargo run --example basic_usage
```

## Prerequisites

### Required Build Tools

- **Rust** (latest stable)
- **CMake** (3.15 or later)
- **C++ compiler** (GCC, Clang, or MSVC)

### macOS
```bash
# Install via Homebrew
brew install cmake

# Or use Xcode command line tools
xcode-select --install
```

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install cmake build-essential
```

## Alternative: System-Installed SDK

If you prefer to use a system-installed Vamp SDK instead of the bundled submodule:

### Building from Source
```bash
git clone https://github.com/vamp-plugins/vamp-plugin-sdk.git
cd vamp-plugin-sdk
cmake -B build -DVAMPSDK_BUILD_SIMPLE_HOST=ON
cmake --build build
sudo cmake --install build
```

### macOS with Homebrew
```bash
brew install vamp-plugin-sdk
```

### Ubuntu/Debian
```bash
sudo apt-get install libvamp-hostsdk3v5 libvamp-sdk2v5 vamp-plugin-sdk
```

The build system automatically detects whether to use the bundled submodule or system installation.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
vamprust = "0.1.0"
```

### Basic Example

```rust
use vamprust::{VampHost, Plugin};

fn main() {
    // Create a host instance
    let host = VampHost::new().expect("Failed to create Vamp host");
    
    // List available plugins
    let plugins = host.list_plugins();
    println!("Found {} plugins", plugins.len());
    
    // Load a plugin
    if let Some(plugin_key) = plugins.first() {
        let mut plugin = Plugin::load(
            &host,
            plugin_key,
            44100.0,  // sample rate
            1024,     // block size
            1024,     // step size  
            1         // channels
        ).expect("Failed to load plugin");
        
        // Process audio (generate sine wave for example)
        let audio_data: Vec<f32> = (0..1024)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 44100.0).sin())
            .collect();
            
        let features = plugin.process_interleaved(&audio_data);
        println!("Extracted {} features", features);
        
        plugin.finish();
    }
}
```

### Running the Example

```bash
cargo run --example basic_usage
```

## Build System

The build system automatically detects and uses the appropriate SDK:

1. **Bundled submodule** (default): Automatically builds the included Vamp SDK
2. **System installation**: Falls back to pkg-config or standard paths
3. **Custom path**: Set `VAMP_SDK_PATH` environment variable

### Build Modes

- **With submodule**: `cargo build` (automatically builds SDK with CMake)
- **System SDK**: Remove/rename the `vamp-plugin-sdk` directory
- **Custom path**: `VAMP_SDK_PATH=/path/to/sdk cargo build`

## Environment Variables

- `VAMP_SDK_PATH`: Path to Vamp SDK installation (overrides auto-detection)
- `VAMP_PATH`: Colon-separated list of directories to search for plugins

## Submodule Management

```bash
# Initialize submodule after cloning
git submodule update --init --recursive

# Update submodule to latest version
git submodule update --remote vamp-plugin-sdk

# Remove submodule (fallback to system SDK)
git submodule deinit vamp-plugin-sdk
```

## Architecture

This wrapper uses:

- `vamp/vamp.h`: The official C plugin API for descriptors and processing
- `vamp-hostsdk/host-c.h`: C-linkage host helpers for discovery and loading
- `bindgen`: Automatic FFI binding generation
- Safe Rust wrappers with proper memory management
- **Bundled SDK**: Git submodule with automatic CMake build

## Platform Support

-  macOS (Intel/ARM - requires cmake and Xcode tools)
-  Linux (tested on Ubuntu/Debian - requires cmake and build-essential)
-  Windows (should work with MSVC but untested)

## License

This project is licensed under the same terms as the Vamp Plugin SDK.