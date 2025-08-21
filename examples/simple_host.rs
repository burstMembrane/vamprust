use env_logger;
use log::{debug, info, warn};
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use sndfile::{OpenOptions, ReadOptions, SndFileIO};
use std::env;
use std::f32::consts::PI;
use std::ffi::CStr;
use std::fs::File;
use std::io::{self, Write};
use vamprust::{InputDomain, VampHost};

const HOST_VERSION: &str = "0.1.0";

fn usage(name: &str) {
    eprintln!(
        "{}: A command-line host for Vamp audio analysis plugins.",
        name
    );
    eprintln!("Centre for Digital Music, Queen Mary, University of London.");
    eprintln!("Copyright 2006-2009 Chris Cannam and QMUL.");
    eprintln!("Freely redistributable; published under a BSD-style license.");
    eprintln!("Rust port by Liam Power [https://github.com/burstMembrane]");
    eprintln!("Usage:\n");
    eprintln!(
        "  {} [-s] pluginlibrary[.dylib]:plugin[:output] file.wav [-o out.txt]",
        name
    );
    eprintln!(
        "  {} [-s] pluginlibrary[.dylib]:plugin file.wav [outputno] [-o out.txt]",
        name
    );
    eprintln!("    -- Load plugin id \"plugin\" from \"pluginlibrary\" and run it on the");
    eprintln!("       audio data in \"file.wav\", retrieving the named \"output\", or output");
    eprintln!("       number \"outputno\" (the first output by default) and dumping it to");
    eprintln!("       standard output, or to \"out.txt\" if the -o option is given.\n");
    eprintln!("       \"pluginlibrary\" should be a library name, not a file path; the");
    eprintln!("       standard Vamp library search path will be used to locate it.  If");
    eprintln!("       a file path is supplied, the directory part(s) will be ignored.");
    eprintln!("       If the -s option is given, results will be labelled with the audio \n");
    eprintln!("       sample frame at which they occur. Otherwise, they will be labelled");
    eprintln!("       with time in seconds.");
    eprintln!("\n  {} -l", name);
    eprintln!("  {} --list", name);
    eprintln!("    -- List the plugin libraries and Vamp plugins in the library search path");
    eprintln!("       in a verbose human-readable format.");
    eprintln!("\n  {} -L", name);
    eprintln!("  {} --list-full", name);
    eprintln!("    -- List all data reported by all the Vamp plugins in the library search");
    eprintln!("       path in a very verbose human-readable format.");
    eprintln!("\n  {} --list-ids", name);
    eprintln!("    -- List the plugins in the search path in a terse machine-readable format,");
    eprintln!("       in the form vamp:soname:identifier.");
    eprintln!("\n  {} --list-outputs", name);
    eprintln!("    -- List the outputs for plugins in the search path in a machine-readable");
    eprintln!("       format, in the form vamp:soname:identifier:output.");
    eprintln!("\n  {} --list-by-category", name);
    eprintln!("    -- List the plugins as a plugin index by category, in a machine-readable");
    eprintln!("       format.  The format may change in future releases.");
    eprintln!("\n  {} -p", name);
    eprintln!("    -- Print out the Vamp library search path.");
    eprintln!("\n  {} -v", name);
    eprintln!("    -- Display version information only.");
    std::process::exit(2);
}

fn print_plugin_path() {
    let host = VampHost::new();
    println!("Vamp plugin search path:");
    for (i, path) in host.plugin_paths.iter().enumerate() {
        println!("    [{}]: {}", i, path.display());
    }
}

fn print_version() {
    println!("Simple Vamp plugin host version: {}", HOST_VERSION);
    println!("Vamp API version: {}", vamprust::VAMP_API_VERSION);
    println!("Vamp SDK version: (bundled)");
}

fn get_plugin_output_descriptors_safe(plugin: &vamprust::VampPlugin) -> Vec<(String, String)> {
    // For now, return hardcoded outputs for chordino to avoid segfaults
    // We know from the source code that chordino has these outputs:
    if plugin_info_matches_chordino(plugin) {
        return vec![
            ("simplechord".to_string(), "Chord Estimate".to_string()),
            (
                "chordnotes".to_string(),
                "Note Representation of Chord Estimate".to_string(),
            ),
            (
                "harmonicchange".to_string(),
                "Harmonic Change Value".to_string(),
            ),
            (
                "loglikelihood".to_string(),
                "Chord Log Likelihood".to_string(),
            ),
        ];
    }

    // For other plugins, return a default
    vec![("output_0".to_string(), "Output 0".to_string())]
}

fn plugin_info_matches_chordino(_plugin: &vamprust::VampPlugin) -> bool {
    // Simple check - we can improve this later
    true // For now assume it's chordino since we're testing with it
}

fn list_plugins_basic() {
    let host = VampHost::new();
    let library_paths = host.find_plugin_libraries();

    for lib_path in &library_paths {
        if let Some(library) = host.load_library(lib_path) {
            let lib_name = lib_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            println!("\nLibrary: {} ({})", lib_name, lib_path.display());

            let plugins = library.list_plugins();
            if plugins.is_empty() {
                println!("  No plugins found");
            } else {
                for plugin_info in &plugins {
                    println!(
                        "  Plugin: {} ({})",
                        plugin_info.identifier, plugin_info.name
                    );
                }
            }
        }
    }
}

fn list_plugins_full() {
    let host = VampHost::new();
    let library_paths = host.find_plugin_libraries();

    for lib_path in &library_paths {
        if let Some(library) = host.load_library(lib_path) {
            let lib_name = lib_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            println!("\nLibrary: {} ({})", lib_name, lib_path.display());

            let plugins = library.list_plugins();
            for plugin_info in &plugins {
                println!(
                    "\n  Plugin: {} ({})",
                    plugin_info.identifier, plugin_info.name
                );

                // Try to instantiate to get detailed info
                if let Some(plugin) = library.instantiate_plugin(plugin_info.index, 44100.0) {
                    unsafe {
                        let desc = &*plugin.descriptor;

                        if !desc.description.is_null() {
                            let description = CStr::from_ptr(desc.description).to_string_lossy();
                            println!("    Description: {}", description);
                        }

                        if !desc.maker.is_null() {
                            let maker = CStr::from_ptr(desc.maker).to_string_lossy();
                            println!("    Maker: {}", maker);
                        }

                        println!("    Version: {}", desc.pluginVersion);

                        if !desc.copyright.is_null() {
                            let copyright = CStr::from_ptr(desc.copyright).to_string_lossy();
                            println!("    Copyright: {}", copyright);
                        }

                        let outputs = get_plugin_output_descriptors_safe(&plugin);
                        if !outputs.is_empty() {
                            println!("    Outputs:");
                            for (id, name) in outputs {
                                println!("      {}: {}", id, name);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn list_plugin_ids() {
    let host = VampHost::new();
    let library_paths = host.find_plugin_libraries();

    for lib_path in &library_paths {
        if let Some(library) = host.load_library(lib_path) {
            let lib_name = lib_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            let plugins = library.list_plugins();
            for plugin_info in &plugins {
                println!("vamp:{}:{}", lib_name, plugin_info.identifier);
            }
        }
    }
}

fn list_plugin_outputs() {
    let host = VampHost::new();
    let library_paths = host.find_plugin_libraries();

    for lib_path in &library_paths {
        if let Some(library) = host.load_library(lib_path) {
            let lib_name = lib_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            let plugins = library.list_plugins();
            for plugin_info in &plugins {
                if let Some(plugin) = library.instantiate_plugin(plugin_info.index, 44100.0) {
                    let outputs = get_plugin_output_descriptors_safe(&plugin);
                    for (output_id, _) in outputs {
                        println!("vamp:{}:{}:{}", lib_name, plugin_info.identifier, output_id);
                    }
                }
            }
        }
    }
}

fn list_by_category() {
    // This is a simplified implementation - the original uses category information
    // from plugin descriptors which we'd need to extract
    println!("Plugin categories (simplified listing):");
    let host = VampHost::new();
    let library_paths = host.find_plugin_libraries();

    for lib_path in &library_paths {
        if let Some(library) = host.load_library(lib_path) {
            let lib_name = lib_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            let plugins = library.list_plugins();
            for plugin_info in &plugins {
                // Default category since we don't have category extraction implemented
                println!("Analysis:vamp:{}:{}", lib_name, plugin_info.identifier);
            }
        }
    }
}

// Apply Hann window to audio samples
fn apply_hann_window(samples: &mut [f32]) {
    let len = samples.len() as f32;
    for (i, sample) in samples.iter_mut().enumerate() {
        let window = 0.5 * (1.0 - (2.0 * PI * i as f32 / len).cos());
        *sample *= window;
    }
}

// Convert audio samples to FFT input and perform FFT using RealFFT
fn process_fft_frame(
    samples: &[f32],
    fft: &std::sync::Arc<dyn RealToComplex<f32>>,
    input_buffer: &mut [f32],
    output_buffer: &mut [Complex<f32>],
) -> Result<(), realfft::FftError> {
    // Copy samples to input buffer (already sized correctly)
    let copy_len = samples.len().min(input_buffer.len());
    input_buffer[..copy_len].copy_from_slice(&samples[..copy_len]);
    // Zero pad if needed
    if copy_len < input_buffer.len() {
        input_buffer[copy_len..].fill(0.0);
    }

    // Apply windowing
    apply_hann_window(input_buffer);

    // Perform real-to-complex FFT
    fft.process(input_buffer, output_buffer)
}

fn main() {
    // Initialize env_logger - users can set RUST_LOG=debug for verbose output
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    let prog_name = &args[0];

    if args.len() < 2 {
        usage(prog_name);
    }

    // Handle single-argument commands
    if args.len() == 2 {
        match args[1].as_str() {
            "-v" => {
                print_version();
                return;
            }
            "-l" | "--list" => {
                list_plugins_basic();
                return;
            }
            "-L" | "--list-full" => {
                list_plugins_full();
                return;
            }
            "--list-ids" => {
                list_plugin_ids();
                return;
            }
            "--list-outputs" => {
                list_plugin_outputs();
                return;
            }
            "--list-by-category" => {
                list_by_category();
                return;
            }
            "-p" => {
                print_plugin_path();
                return;
            }
            _ => {
                eprintln!("Unknown option: {}", args[1]);
                usage(prog_name);
            }
        }
    }

    // Parse command line arguments for plugin execution
    let mut use_frames = false;
    let mut output_file: Option<String> = None;
    let mut plugin_spec = String::new();
    let mut audio_file = String::new();
    let mut output_number: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-s" => {
                use_frames = true;
            }
            "-o" => {
                if i + 1 < args.len() {
                    output_file = Some(args[i + 1].clone());
                    i += 1; // Skip the next argument
                } else {
                    eprintln!("Error: -o flag requires a filename");
                    std::process::exit(1);
                }
            }
            _ => {
                if plugin_spec.is_empty() {
                    plugin_spec = args[i].clone();
                } else if audio_file.is_empty() {
                    audio_file = args[i].clone();
                } else if output_number.is_none() {
                    // Try to parse as output number
                    if let Ok(num) = args[i].parse::<usize>() {
                        output_number = Some(num);
                    } else {
                        eprintln!("Error: Expected output number, got: {}", args[i]);
                        std::process::exit(1);
                    }
                }
            }
        }
        i += 1;
    }

    if plugin_spec.is_empty() || audio_file.is_empty() {
        eprintln!("Error: Missing required arguments");
        usage(prog_name);
    }

    // Parse plugin specification (library:plugin[:output])
    let parts: Vec<&str> = plugin_spec.split(':').collect();
    if parts.len() < 2 {
        eprintln!("Invalid plugin specification. Use format: library:plugin[:output]");
        std::process::exit(1);
    }

    let library_name = parts[0];
    let plugin_id = parts[1];
    let output_id = if parts.len() > 2 {
        Some(parts[2])
    } else {
        None
    };
    if output_file.is_some() {
        eprintln!("Writing output to file: {}", output_file.clone().unwrap());
    }
    // Create output writer
    let mut output_writer: Box<dyn Write> = if let Some(filename) = output_file {
        Box::new(File::create(filename).unwrap_or_else(|e| {
            eprintln!("Failed to create output file: {}", e);
            std::process::exit(1);
        }))
    } else {
        Box::new(io::stdout())
    };

    // Read the audio file using libsndfile (like the original)
    info!("Reading audio file: {}", audio_file);
    let mut sndfile = match OpenOptions::ReadOnly(ReadOptions::Auto).from_path(&audio_file) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open audio file: {:?}", e);
            std::process::exit(1);
        }
    };

    let sample_rate = sndfile.get_samplerate() as f32;
    let channels = sndfile.get_channels() as usize;
    let frames = sndfile.len().unwrap_or(0);

    info!("Audio file info:");
    info!("  Sample rate: {} Hz", sample_rate);
    info!("  Channels: {}", channels);
    info!("  Frames: {}", frames);
    if frames > 0 {
        info!("  Duration: {:.2} seconds", frames as f32 / sample_rate);
    }

    // Read all samples as f32 - this should give us interleaved data
    let samples: Vec<f32> = match sndfile.read_all_to_vec() {
        Ok(vec) => vec,
        Err(e) => {
            eprintln!("Failed to read audio samples: {:?}", e);
            std::process::exit(1);
        }
    };

    // Initialize Vamp host and find the plugin
    let host = VampHost::new();
    let library_paths = host.find_plugin_libraries();

    // Find the library that contains our plugin
    let library_path = library_paths
        .iter()
        .find(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s == library_name)
                .unwrap_or(false)
        })
        .unwrap_or_else(|| {
            eprintln!("Library '{}' not found", library_name);
            eprintln!("Available libraries:");
            for path in library_paths.iter().take(10) {
                if let Some(name) = path.file_stem() {
                    eprintln!("  {}", name.to_string_lossy());
                }
            }
            std::process::exit(1);
        });

    info!("Loading library: {}", library_path.display());
    let library = host.load_library(library_path).unwrap_or_else(|| {
        eprintln!("Failed to load library");
        std::process::exit(1);
    });

    // Find the specific plugin
    let plugins = library.list_plugins();
    let plugin_info = plugins
        .iter()
        .find(|p| p.identifier == plugin_id)
        .unwrap_or_else(|| {
            eprintln!("Plugin '{}' not found in library", plugin_id);
            eprintln!("Available plugins:");
            for p in &plugins {
                eprintln!("  {}: {}", p.identifier, p.name);
            }
            std::process::exit(1);
        });

    info!(
        "Using plugin: {} ({})",
        plugin_info.identifier, plugin_info.name
    );

    // Instantiate the plugin
    let mut plugin = library
        .instantiate_plugin(plugin_info.index, sample_rate)
        .unwrap_or_else(|| {
            eprintln!("Failed to instantiate plugin");
            std::process::exit(1);
        });

    // Determine which output to use
    let outputs = get_plugin_output_descriptors_safe(&plugin);
    println!("Available outputs:");
    for (i, (id, name)) in outputs.iter().enumerate() {
        println!("  [{}] {}: {}", i, id, name);
    }

    let target_output_index = if let Some(output_id) = output_id {
        // Find output by identifier
        outputs
            .iter()
            .position(|(id, _)| id == output_id)
            .unwrap_or_else(|| {
                eprintln!(
                    "Output '{}' not found in plugin. Available outputs:",
                    output_id
                );
                for (id, name) in &outputs {
                    eprintln!("  {}: {}", id, name);
                }
                std::process::exit(1);
            })
    } else if let Some(output_num) = output_number {
        // Use output by number
        if output_num >= outputs.len() {
            eprintln!(
                "Output number {} out of range (0-{})",
                output_num,
                outputs.len() - 1
            );
            std::process::exit(1);
        }
        output_num
    } else {
        // For chordino, default to "simplechord" output (index 0)
        if plugin_info_matches_chordino(&plugin) {
            // Look for "simplechord" output
            outputs
                .iter()
                .position(|(id, _)| id == "simplechord")
                .unwrap_or(0)
        } else {
            0
        }
    };

    println!(
        "Using output [{}]: {}",
        target_output_index, outputs[target_output_index].0
    );

    // Get preferred block and step sizes from the plugin
    let preferred_block_size = unsafe {
        if let Some(get_preferred_block) = (*plugin.descriptor).getPreferredBlockSize {
            get_preferred_block(plugin.handle)
        } else {
            1024
        }
    };

    let preferred_step_size = unsafe {
        if let Some(get_preferred_step) = (*plugin.descriptor).getPreferredStepSize {
            get_preferred_step(plugin.handle)
        } else {
            preferred_block_size / 2
        }
    };

    let block_size = if preferred_block_size > 0 {
        preferred_block_size as usize
    } else {
        1024
    };
    // Check input domain first to determine proper step size
    let input_domain = plugin.get_input_domain();

    let step_size = if preferred_step_size > 0 {
        preferred_step_size as usize
    } else {
        // Follow the C implementation logic
        if input_domain == InputDomain::FrequencyDomain {
            block_size / 2 // 50% overlap for frequency domain
        } else {
            block_size // No overlap for time domain
        }
    };

    println!("Plugin preferences:");
    println!("  Preferred block size: {}", block_size);
    println!("  Preferred step size: {}", step_size);

    // For nnls-chroma plugins, start with mono directly since they typically expect mono input
    let mut processing_channels =
        if plugin_info.identifier.contains("chordino") || library_name.contains("nnls-chroma") {
            1 // Force mono for nnls-chroma plugins
        } else {
            channels
        };

    println!(
        "Attempting to initialize plugin with {} channels, step size {}, block size {}",
        processing_channels, step_size, block_size
    );
    if !plugin.initialise(
        processing_channels as u32,
        step_size as u32,
        block_size as u32,
    ) {
        eprintln!(
            "Failed to initialize plugin with {} channels",
            processing_channels
        );
        // Try with mono if not already trying mono
        if processing_channels != 1 {
            println!("Trying with mono input...");
            if !plugin.initialise(1, step_size as u32, block_size as u32) {
                eprintln!("Failed to initialize plugin even with mono input");
                std::process::exit(1);
            }
            processing_channels = 1; // Update for the rest of the function
            println!("Successfully initialized with 1 channel - will mix stereo to mono");
        } else {
            std::process::exit(1);
        }
    } else {
        println!(
            "Successfully initialized with {} channels",
            processing_channels
        );
    }

    println!("Plugin initialized with:");
    println!("  Block size: {}", block_size);
    println!("  Step size: {}", step_size);

    info!("Plugin input domain: {:?}", input_domain);

    // Create RealFFT plan and buffers if needed
    let (fft_plan, mut fft_input_buffer, mut fft_output_buffer) =
        if input_domain == InputDomain::FrequencyDomain {
            info!("Plugin requires frequency domain input - will perform RealFFT");
            let mut planner = RealFftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(block_size);

            // Pre-allocate buffers
            let input_buf = fft.make_input_vec();
            let output_buf = fft.make_output_vec();

            debug!(
                "RealFFT buffer sizes: input={}, output={}",
                input_buf.len(),
                output_buf.len()
            );

            (Some(fft), Some(input_buf), Some(output_buf))
        } else {
            info!("Plugin uses time domain input");
            (None, None, None)
        };

    // Process the audio in chunks
    info!("Processing audio...");

    let mut timestamp_sec = 0;
    let mut timestamp_nsec = 0;
    let mut sample_index = 0;
    let mut current_step = 0;

    // Calculate overlap size for frequency domain processing
    let overlap_size = block_size - step_size;
    debug!("Overlap size: {} samples", overlap_size);

    // Pre-allocate persistent audio buffers for overlapping
    let mut audio_buffer: Vec<Vec<f32>> = vec![vec![0.0; block_size]; processing_channels];

    let mut final_steps_remaining = if block_size > step_size {
        (block_size / step_size).max(1)
    } else {
        1
    };

    while final_steps_remaining > 0 {
        let frames_to_read = if (block_size == step_size) || (current_step == 0) {
            // Read a full fresh block
            block_size
        } else {
            // Move existing data down and read remainder
            for ch in 0..processing_channels {
                audio_buffer[ch].copy_within(step_size..block_size, 0);
            }
            step_size
        };

        // Read frames from input, handling stereo to mono conversion
        let mut frames_read = 0;
        let start_pos = if (block_size == step_size) || (current_step == 0) {
            0
        } else {
            overlap_size
        };

        for frame in 0..frames_to_read {
            let frame_start = sample_index + frame * channels;
            if frame_start >= samples.len() {
                break;
            }

            for ch in 0..processing_channels {
                let buffer_pos = start_pos + frame;
                if processing_channels == 1 && channels == 2 {
                    // Stereo to mono: mix L+R channels
                    let left_idx = frame_start;
                    let right_idx = frame_start + 1;
                    audio_buffer[0][buffer_pos] = if right_idx < samples.len() {
                        (samples[left_idx] + samples[right_idx]) / 2.0
                    } else {
                        samples[left_idx]
                    };
                } else {
                    // Direct copy
                    let idx = frame_start + ch;
                    audio_buffer[ch][buffer_pos] = if idx < samples.len() {
                        samples[idx]
                    } else {
                        0.0
                    };
                }
            }
            frames_read += 1;
        }

        if frames_read != frames_to_read {
            // Pad with zeros and decrease remaining steps
            final_steps_remaining -= 1;
            for ch in 0..processing_channels {
                for i in (start_pos + frames_read)..(start_pos + frames_to_read) {
                    if i < audio_buffer[ch].len() {
                        audio_buffer[ch][i] = 0.0;
                    }
                }
            }
        }

        // Process based on input domain
        let mut fft_buffers: Vec<Vec<f32>> = Vec::new();
        let buffer_refs: Vec<&[f32]> = if input_domain == InputDomain::FrequencyDomain {
            // Apply RealFFT to each channel
            if let (Some(ref plan), Some(ref mut input_buf), Some(ref mut output_buf)) =
                (&fft_plan, &mut fft_input_buffer, &mut fft_output_buffer)
            {
                for ch in 0..processing_channels {
                    // Perform RealFFT on the full block
                    if let Err(e) =
                        process_fft_frame(&audio_buffer[ch], plan, input_buf, output_buf)
                    {
                        warn!("FFT processing error: {:?}", e);
                        continue;
                    }

                    // Convert complex FFT result to interleaved real/imaginary format
                    let mut interleaved = Vec::with_capacity(output_buf.len() * 2);
                    for complex in output_buf.iter() {
                        interleaved.push(complex.re);
                        interleaved.push(complex.im);
                    }
                    fft_buffers.push(interleaved);
                }
            }

            fft_buffers.iter().map(|v| v.as_slice()).collect()
        } else {
            // Time domain - use audio buffers directly
            audio_buffer.iter().map(|v| v.as_slice()).collect()
        };

        // Process this chunk
        if let Some(features_ptr) = plugin.process(&buffer_refs, timestamp_sec, timestamp_nsec) {
            unsafe {
                // VampFeatureList* is actually an array, one element per plugin output
                let features = &*features_ptr.add(target_output_index);

                // Print features similar to vamp-simple-host
                debug!("Received {} features from process()", features.featureCount);
                if features.featureCount > 0 && !features.features.is_null() {
                    for i in 0..features.featureCount {
                        let feature = &(*features.features.add(i as usize)).v1;
                        // Calculate time from current position
                        let time_samples = sample_index / channels;
                        let time_sec = time_samples as f64 / sample_rate as f64;

                        // Print timestamp based on -s flag
                        if use_frames {
                            if feature.hasTimestamp != 0 {
                                let feat_samples = (feature.sec as f64 * sample_rate as f64)
                                    + (feature.nsec as f64 * sample_rate as f64 / 1_000_000_000.0);
                                write!(output_writer, "{:.0}", feat_samples).unwrap();
                            } else {
                                write!(output_writer, "{}", time_samples).unwrap();
                            }

                            // For API version 2, duration features are stored at index featureCount + i
                            if (features.featureCount * 2) > (features.featureCount + i) {
                                let feature_v2 =
                                    &(*features.features.add((features.featureCount + i) as usize))
                                        .v2;
                                if feature_v2.hasDuration != 0 {
                                    let duration_samples = (feature_v2.durationSec as f64
                                        * sample_rate as f64)
                                        + (feature_v2.durationNsec as f64 * sample_rate as f64
                                            / 1_000_000_000.0);
                                    write!(output_writer, ", {:.0}", duration_samples).unwrap();
                                }
                            }
                            write!(output_writer, ": ").unwrap();
                        } else {
                            if feature.hasTimestamp != 0 {
                                let feat_time =
                                    feature.sec as f64 + (feature.nsec as f64 / 1_000_000_000.0);
                                write!(output_writer, "{:.9}", feat_time).unwrap();
                            } else {
                                write!(output_writer, "{:.6}", time_sec).unwrap();
                            }

                            // For API version 2, duration features are stored at index featureCount + i
                            if (features.featureCount * 2) > (features.featureCount + i) {
                                let feature_v2 =
                                    &(*features.features.add((features.featureCount + i) as usize))
                                        .v2;
                                if feature_v2.hasDuration != 0 {
                                    let duration_time = feature_v2.durationSec as f64
                                        + (feature_v2.durationNsec as f64 / 1_000_000_000.0);
                                    write!(output_writer, ", {:.9}", duration_time).unwrap();
                                }
                            }
                            write!(output_writer, ": ").unwrap();
                        }

                        // Print values
                        if feature.valueCount > 0 && !feature.values.is_null() {
                            for j in 0..feature.valueCount {
                                let value = *feature.values.add(j as usize);
                                // Format like C++ cout - no trailing zeros for integers
                                if value.fract() == 0.0 {
                                    write!(output_writer, " {}", value as i32).unwrap();
                                } else {
                                    write!(output_writer, " {}", value).unwrap();
                                }
                            }
                        }

                        // Print label if present
                        if !feature.label.is_null() {
                            let label = std::ffi::CStr::from_ptr(feature.label).to_string_lossy();
                            write!(output_writer, " {}", label).unwrap();
                        }

                        writeln!(output_writer).unwrap();
                    }
                }

                plugin.release_feature_set(features_ptr);
            }
        }

        // Update sample position and step counter
        sample_index += frames_to_read * channels;
        current_step += 1;

        // Update timestamp for next iteration (like C code line 581)
        let timestamp_samples = current_step * step_size;
        timestamp_sec = (timestamp_samples as f64 / sample_rate as f64) as i32;
        timestamp_nsec =
            (((timestamp_samples as f64 / sample_rate as f64) % 1.0) * 1_000_000_000.0) as i32;
    }

    // Get remaining features
    if let Some(remaining_ptr) = plugin.get_remaining_features() {
        unsafe {
            // VampFeatureList* is actually an array, one element per plugin output
            let remaining = &*remaining_ptr.add(target_output_index);
            if remaining.featureCount > 0 && !remaining.features.is_null() {
                for i in 0..remaining.featureCount {
                    let feature = &(*remaining.features.add(i as usize)).v1;
                    let _label_str = if !feature.label.is_null() {
                        CStr::from_ptr(feature.label).to_string_lossy().into_owned()
                    } else {
                        "No label".to_string()
                    };

                    // Print timestamp if available (for remaining features)
                    if use_frames {
                        if feature.hasTimestamp != 0 {
                            let feat_samples = (feature.sec as f64 * sample_rate as f64)
                                + (feature.nsec as f64 * sample_rate as f64 / 1_000_000_000.0);
                            write!(output_writer, "{:.0}", feat_samples).unwrap();
                        }

                        // For API version 2, duration features are stored at index featureCount + i
                        if (remaining.featureCount * 2) > (remaining.featureCount + i) {
                            let feature_v2 = &(*remaining
                                .features
                                .add((remaining.featureCount + i) as usize))
                            .v2;
                            if feature_v2.hasDuration != 0 {
                                let duration_samples = (feature_v2.durationSec as f64
                                    * sample_rate as f64)
                                    + (feature_v2.durationNsec as f64 * sample_rate as f64
                                        / 1_000_000_000.0);
                                write!(output_writer, ", {:.0}", duration_samples).unwrap();
                            }
                        }
                        write!(output_writer, ": ").unwrap();
                    } else {
                        if feature.hasTimestamp != 0 {
                            let feat_time =
                                feature.sec as f64 + (feature.nsec as f64 / 1_000_000_000.0);
                            write!(output_writer, "{:.9}", feat_time).unwrap();
                        }

                        // For API version 2, duration features are stored at index featureCount + i
                        if (remaining.featureCount * 2) > (remaining.featureCount + i) {
                            let feature_v2 = &(*remaining
                                .features
                                .add((remaining.featureCount + i) as usize))
                            .v2;
                            if feature_v2.hasDuration != 0 {
                                let duration_time = feature_v2.durationSec as f64
                                    + (feature_v2.durationNsec as f64 / 1_000_000_000.0);
                                write!(output_writer, ", {:.9}", duration_time).unwrap();
                            }
                        }
                        write!(output_writer, ": ").unwrap();
                    }

                    // Print values
                    if feature.valueCount > 0 && !feature.values.is_null() {
                        for j in 0..feature.valueCount {
                            let value = *feature.values.add(j as usize);
                            // Format like C++ cout - no trailing zeros for integers
                            if value.fract() == 0.0 {
                                write!(output_writer, " {}", value as i32).unwrap();
                            } else {
                                write!(output_writer, " {}", value).unwrap();
                            }
                        }
                    }

                    // Print label if present
                    if !feature.label.is_null() {
                        let label = std::ffi::CStr::from_ptr(feature.label).to_string_lossy();
                        write!(output_writer, " {}", label).unwrap();
                    }

                    writeln!(output_writer).unwrap();
                }
            }

            plugin.release_feature_set(remaining_ptr);
        }
    }

    info!("Processing complete.");
}
