use vamprust::VampHost;

fn main() {
    println!("Vamp Plugin SDK Rust Wrapper Example");
    println!("====================================");

    let host = VampHost::new();

    println!("Scanning for plugin libraries...");
    let library_paths = host.find_plugin_libraries();

    if library_paths.is_empty() {
        println!("No plugin libraries found. Make sure Vamp plugins are installed and VAMP_PATH is set correctly.");
        return;
    }

    println!("Found {} plugin libraries:", library_paths.len());
    for (i, lib_path) in library_paths.iter().enumerate().take(10) {
        println!("  {}: {}", i + 1, lib_path.display());
    }

    // Try to find a simpler plugin library (like vamp-example-plugins)
    let example_lib = library_paths
        .iter()
        .find(|p| p.to_string_lossy().contains("vamp-example-plugins"))
        .or_else(|| library_paths.first());

    if let Some(first_lib_path) = example_lib {
        println!("\nLoading library: {}", first_lib_path.display());

        if let Some(library) = host.load_library(first_lib_path) {
            println!("Successfully loaded library!");

            let plugins = library.list_plugins();
            println!("Found {} plugins in library:", plugins.len());

            for plugin_info in plugins.iter().take(5) {
                println!("  {}: {}", plugin_info.identifier, plugin_info.name);
            }

            if let Some(first_plugin) = plugins.first() {
                println!(
                    "\nAttempting to instantiate plugin: {}",
                    first_plugin.identifier
                );

                let sample_rate = 44100.0;

                if let Some(mut plugin) =
                    library.instantiate_plugin(first_plugin.index, sample_rate)
                {
                    println!("Successfully instantiated plugin!");

                    let channels = 1;
                    let step_size = 1024;
                    let block_size = 1024;

                    if plugin.initialise(channels, step_size, block_size) {
                        println!("Plugin initialized successfully!");
                        println!("  Sample rate: {}", sample_rate);
                        println!("  Channels: {}", channels);
                        println!("  Step size: {}", step_size);
                        println!("  Block size: {}", block_size);

                        // Generate test audio data (440Hz sine wave)
                        let audio_data: Vec<f32> = (0..block_size)
                            .map(|i| {
                                (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate).sin()
                                    * 0.5
                            })
                            .collect();

                        println!("\nProcessing {} samples of audio...", audio_data.len());

                        // Create input buffer array
                        let input_buffers = vec![audio_data.as_slice()];

                        // Process with timestamp (0 seconds, 0 nanoseconds)
                        if let Some(features_ptr) = plugin.process(&input_buffers, 0, 0) {
                            unsafe {
                                let features = &*features_ptr;
                                println!("Extracted {} feature(s)", features.featureCount);
                                // Must release the feature set to avoid memory issues
                                plugin.release_feature_set(features_ptr);
                            }
                        } else {
                            println!("No features returned from processing");
                        }

                        // Get any remaining features
                        if let Some(remaining_ptr) = plugin.get_remaining_features() {
                            unsafe {
                                let remaining = &*remaining_ptr;
                                println!("Got {} remaining feature(s)", remaining.featureCount);
                                // Must release the feature set
                                plugin.release_feature_set(remaining_ptr);
                            }
                        } else {
                            println!("No remaining features");
                        }

                        println!("Plugin processing completed successfully!");
                    } else {
                        println!("Failed to initialize plugin");
                    }
                } else {
                    println!("Failed to instantiate plugin: {}", first_plugin.identifier);
                }
            }
        } else {
            println!("Failed to load library: {}", first_lib_path.display());
        }
    }
}
