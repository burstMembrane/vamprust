#!/usr/bin/env python3
"""
Basic usage example for VampRust Python bindings.

This example demonstrates:
1. Creating a VampHost
2. Discovering available plugins
3. Loading and initializing a plugin
4. Processing audio data
5. Extracting features
"""

import numpy as np
import vamprust

def main():
    print("VampRust Python Bindings - Basic Usage Example")
    print("=" * 50)
    
    # Create a host instance
    host = vamprust.VampHost()
    print(f"Created VampHost: {host}")
    
    # Find available plugin libraries
    print("\nScanning for plugin libraries...")
    library_paths = host.find_plugin_libraries()
    
    if not library_paths:
        print("No plugin libraries found. Make sure Vamp plugins are installed.")
        print("On macOS: brew install vamp-plugin-sdk")
        print("On Ubuntu: sudo apt-get install vamp-plugin-sdk")
        return
    
    print(f"Found {len(library_paths)} plugin libraries:")
    for i, lib_path in enumerate(library_paths[:5]):  # Show first 5
        print(f"  {i+1}: {lib_path}")
    
    # Load the first library and list its plugins
    first_lib_path = library_paths[0]
    print(f"\nLoading library: {first_lib_path}")
    
    library = host.load_library(first_lib_path)
    if not library:
        print("Failed to load library")
        return
    
    print(f"Successfully loaded library: {library}")
    
    # List plugins in the library
    plugins = library.list_plugins()
    print(f"\nFound {len(plugins)} plugins in library:")
    for plugin_info in plugins[:3]:  # Show first 3
        print(f"  - {plugin_info.identifier}: {plugin_info.name}")
    
    if not plugins:
        print("No plugins found in library")
        return
    
    # Instantiate the first plugin
    first_plugin = plugins[0]
    print(f"\nInstantiating plugin: {first_plugin.identifier}")
    
    sample_rate = 44100.0
    plugin = library.instantiate_plugin(first_plugin.index, sample_rate)
    
    if not plugin:
        print("Failed to instantiate plugin")
        return
    
    print(f"Successfully instantiated plugin: {plugin}")
    
    # Get plugin information
    print(f"Input domain: {plugin.get_input_domain()}")
    print(f"Preferred block size: {plugin.get_preferred_block_size()}")
    print(f"Preferred step size: {plugin.get_preferred_step_size()}")
    
    # Initialize the plugin
    channels = 1
    block_size = plugin.get_preferred_block_size()
    step_size = plugin.get_preferred_step_size()
    if step_size == 0:
        step_size = block_size // 2
    
    print(f"\nInitializing plugin with:")
    print(f"  Channels: {channels}")
    print(f"  Block size: {block_size}")
    print(f"  Step size: {step_size}")
    
    if not plugin.initialise(channels, step_size, block_size):
        print("Failed to initialize plugin")
        return
    
    print("Plugin initialized successfully!")
    
    # Generate test audio data (440Hz sine wave)
    duration = 1.0  # 1 second
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, False)
    frequency = 440.0  # A4 note
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    print(f"\nProcessing {samples} samples of 440Hz sine wave...")
    
    # Process audio in blocks
    all_features = []
    
    for i in range(0, samples - block_size + 1, step_size):
        block = audio_data[i:i + block_size]
        input_buffers = [block.tolist()]  # Single channel
        
        # Calculate timestamp
        sec = int(i / sample_rate)
        nsec = int((i / sample_rate - sec) * 1e9)
        
        # Process block
        features = plugin.process(input_buffers, sec, nsec)
        if features and features['features']:
            all_features.extend(features['features'])
            print(f"  Block {i//step_size + 1}: {features['feature_count']} features")
    
    # Get remaining features
    remaining = plugin.get_remaining_features()
    if remaining and remaining['features']:
        all_features.extend(remaining['features'])
        print(f"  Remaining: {remaining['feature_count']} features")
    
    print(f"\nTotal features extracted: {len(all_features)}")
    
    # Show some feature information
    if all_features:
        first_feature = all_features[0]
        print(f"\nFirst feature:")
        print(f"  Has timestamp: {first_feature.get('has_timestamp', False)}")
        if first_feature.get('has_timestamp'):
            timestamp = first_feature.get('sec', 0) + first_feature.get('nsec', 0) * 1e-9
            print(f"  Timestamp: {timestamp:.6f} seconds")
        print(f"  Values: {first_feature.get('values', [])[:5]}...")  # Show first 5 values
        if first_feature.get('label'):
            print(f"  Label: {first_feature['label']}")
    
    print("\nProcessing completed successfully!")

if __name__ == "__main__":
    main()