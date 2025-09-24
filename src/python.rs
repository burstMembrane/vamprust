use crate::{InputDomain, PluginInfo, VampHost, VampLibrary, VampPlugin};
#[cfg(feature = "python")]
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
#[pyclass]
#[derive(Clone)]
pub struct PyVampHost {
    host: VampHost,
}

#[pymethods]
impl PyVampHost {
    #[new]
    fn new() -> Self {
        PyVampHost {
            host: VampHost::new(),
        }
    }

    fn find_plugin_libraries(&self) -> Vec<String> {
        self.host
            .find_plugin_libraries()
            .into_iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect()
    }

    fn load_library(&self, library_path: String) -> PyResult<Option<PyVampLibrary>> {
        log::debug!("PyVampHost: Attempting to load library: {}", library_path);
        match self.host.load_library(&library_path) {
            Some(library) => {
                log::debug!("PyVampHost: Successfully loaded library: {}", library_path);
                Ok(Some(PyVampLibrary { library }))
            },
            None => {
                log::warn!("PyVampHost: Failed to load library: {}", library_path);
                Ok(None)
            },
        }
    }

    fn __repr__(&self) -> String {
        format!("VampHost(plugin_paths={:?})", self.host.plugin_paths)
    }

    #[pyo3(signature = (plugin_id, audio_samples, sample_rate, channels=1, output_index=None))]
    fn process_audio_auto(
        &self,
        plugin_id: String,
        audio_samples: Vec<f32>,
        sample_rate: f32,
        channels: Option<usize>,
        output_index: Option<usize>,
    ) -> PyResult<Option<PyObject>> {
        // Automatically find and load the plugin, then process audio
        let libraries = self.host.find_plugin_libraries();
        log::debug!("PyVampHost: Searching for plugin '{}' in {} libraries", plugin_id, libraries.len());

        for library_path in libraries {
            log::debug!("PyVampHost: Checking library: {}", library_path.display());
            if let Some(library) = self.host.load_library(&library_path) {
                log::debug!("PyVampHost: Successfully loaded library, listing plugins");
                let plugins = library.list_plugins();
                log::debug!("PyVampHost: Found {} plugins in library", plugins.len());

                for (index, plugin_info) in plugins.iter().enumerate() {
                    log::debug!("PyVampHost: Checking plugin {}: '{}'", index, plugin_info.identifier);
                    if plugin_info.identifier == plugin_id {
                        log::debug!("PyVampHost: Found target plugin '{}' at index {}", plugin_id, index);
                        // Found the plugin! Instantiate and process
                        if let Some(mut plugin) = library.instantiate_plugin(index, sample_rate) {
                            // Initialize the plugin
                            let processing_channels = channels.unwrap_or(1);

                            // Get preferred sizes - add null checks for safety
                            let preferred_block_size = unsafe {
                                if !plugin.descriptor.is_null() && 
                                   (*plugin.descriptor).getPreferredBlockSize.is_some() {
                                    if let Some(get_preferred_block) =
                                        (*plugin.descriptor).getPreferredBlockSize
                                    {
                                        get_preferred_block(plugin.handle)
                                    } else {
                                        1024
                                    }
                                } else {
                                    1024
                                }
                            };

                            let preferred_step_size = unsafe {
                                if !plugin.descriptor.is_null() && 
                                   (*plugin.descriptor).getPreferredStepSize.is_some() {
                                    if let Some(get_preferred_step) =
                                        (*plugin.descriptor).getPreferredStepSize
                                    {
                                        get_preferred_step(plugin.handle)
                                    } else {
                                        preferred_block_size / 2
                                    }
                                } else {
                                    preferred_block_size / 2
                                }
                            };

                            let block_size = if preferred_block_size > 0 {
                                preferred_block_size as usize
                            } else {
                                1024
                            };

                            let input_domain = plugin.get_input_domain();

                            let step_size = if preferred_step_size > 0 {
                                preferred_step_size as usize
                            } else {
                                if input_domain == crate::InputDomain::FrequencyDomain {
                                    block_size / 2
                                } else {
                                    block_size
                                }
                            };

                            // Initialize the plugin
                            if !plugin.initialise(
                                processing_channels as u32,
                                step_size as u32,
                                block_size as u32,
                            ) {
                                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                                    "Failed to initialize plugin",
                                ));
                            }

                            // Now create a PyVampPlugin and use its process_audio_full method
                            let mut py_plugin = PyVampPlugin {
                                plugin: std::mem::ManuallyDrop::new(plugin),
                                initialized: true,
                                block_size: Some(block_size),
                                step_size: Some(step_size),
                                fft_plan: None,
                                fft_input_buf: None,
                                fft_output_buf: None,
                                fft_scratch: None,
                                fft_interleaved: None,
                                hann_window: None,
                                fft_block_size: None,
                            };

                            return py_plugin.process_audio_full(
                                audio_samples,
                                sample_rate,
                                processing_channels,
                                output_index,
                            );
                        }
                    }
                }
            }
        }

        Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Plugin '{}' not found",
            plugin_id
        )))
    }

    #[cfg(feature = "python")]
    #[pyo3(signature = (plugin_id, audio, sample_rate, channels=None, output_index=None))]
    fn process_audio_auto_nd(
        &self,
        py: Python<'_>,
        plugin_id: String,
        audio: Bound<'_, PyAny>,
        sample_rate: f32,
        channels: Option<usize>,
        output_index: Option<usize>,
    ) -> PyResult<Option<PyObject>> {
        // Automatically find and load the plugin, then process audio with zero-copy NumPy
        let libraries = self.host.find_plugin_libraries();

        for library_path in libraries {
            if let Some(library) = self.host.load_library(&library_path) {
                let plugins = library.list_plugins();

                for (index, plugin_info) in plugins.iter().enumerate() {
                    if plugin_info.identifier == plugin_id {
                        // Found the plugin! Instantiate and process
                        if let Some(mut plugin) = library.instantiate_plugin(index, sample_rate) {
                            // Initialize the plugin
                            let processing_channels = channels.unwrap_or(1);

                            // Get preferred sizes - add null checks for safety
                            let preferred_block_size = unsafe {
                                if !plugin.descriptor.is_null() && 
                                   (*plugin.descriptor).getPreferredBlockSize.is_some() {
                                    if let Some(get_preferred_block) =
                                        (*plugin.descriptor).getPreferredBlockSize
                                    {
                                        get_preferred_block(plugin.handle)
                                    } else {
                                        1024
                                    }
                                } else {
                                    1024
                                }
                            };

                            let preferred_step_size = unsafe {
                                if !plugin.descriptor.is_null() && 
                                   (*plugin.descriptor).getPreferredStepSize.is_some() {
                                    if let Some(get_preferred_step) =
                                        (*plugin.descriptor).getPreferredStepSize
                                    {
                                        get_preferred_step(plugin.handle)
                                    } else {
                                        preferred_block_size / 2
                                    }
                                } else {
                                    preferred_block_size / 2
                                }
                            };

                            let block_size = if preferred_block_size > 0 {
                                preferred_block_size as usize
                            } else {
                                1024
                            };

                            let input_domain = plugin.get_input_domain();

                            let step_size = if preferred_step_size > 0 {
                                preferred_step_size as usize
                            } else {
                                if input_domain == crate::InputDomain::FrequencyDomain {
                                    block_size / 2
                                } else {
                                    block_size
                                }
                            };

                            // Initialize the plugin
                            if !plugin.initialise(
                                processing_channels as u32,
                                step_size as u32,
                                block_size as u32,
                            ) {
                                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                                    "Failed to initialize plugin",
                                ));
                            }

                            // Now create a PyVampPlugin and use its process_audio_nd method
                            let mut py_plugin = PyVampPlugin {
                                plugin: std::mem::ManuallyDrop::new(plugin),
                                initialized: true,
                                block_size: Some(block_size),
                                step_size: Some(step_size),
                                fft_plan: None,
                                fft_input_buf: None,
                                fft_output_buf: None,
                                fft_scratch: None,
                                fft_interleaved: None,
                                hann_window: None,
                                fft_block_size: None,
                            };

                            return py_plugin.process_audio_nd(
                                py,
                                audio,
                                sample_rate,
                                Some(processing_channels),
                                output_index,
                            );
                        }
                    }
                }
            }
        }

        Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Plugin '{}' not found",
            plugin_id
        )))
    }
}

#[pyclass(unsendable)]
pub struct PyVampLibrary {
    library: VampLibrary,
}

#[pymethods]
impl PyVampLibrary {
    fn list_plugins(&self) -> Vec<PyPluginInfo> {
        self.library
            .list_plugins()
            .into_iter()
            .map(|info| PyPluginInfo { info })
            .collect()
    }

    fn instantiate_plugin(
        &self,
        plugin_index: usize,
        sample_rate: f32,
    ) -> PyResult<Option<PyVampPlugin>> {
        match self.library.instantiate_plugin(plugin_index, sample_rate) {
            Some(plugin) => Ok(Some(PyVampPlugin {
                plugin: std::mem::ManuallyDrop::new(plugin),
                initialized: false,
                block_size: None,
                step_size: None,
                fft_plan: None,
                fft_input_buf: None,
                fft_output_buf: None,
                fft_scratch: None,
                fft_interleaved: None,
                hann_window: None,
                fft_block_size: None,
            })),
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        let plugins = self.library.list_plugins();
        format!("VampLibrary(plugins={} found)", plugins.len())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyPluginInfo {
    info: PluginInfo,
}

#[pymethods]
impl PyPluginInfo {
    #[getter]
    fn identifier(&self) -> String {
        self.info.identifier.clone()
    }

    #[getter]
    fn name(&self) -> String {
        self.info.name.clone()
    }

    #[getter]
    fn index(&self) -> usize {
        self.info.index
    }

    #[getter]
    fn library_path(&self) -> String {
        self.info.library_path.to_string_lossy().to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "PluginInfo(identifier='{}', name='{}')",
            self.info.identifier, self.info.name
        )
    }
}

#[pyclass(unsendable)]
pub struct PyVampPlugin {
    plugin: std::mem::ManuallyDrop<VampPlugin>,
    initialized: bool,
    block_size: Option<usize>,
    step_size: Option<usize>,
    // FFT caching
    fft_plan: Option<std::sync::Arc<dyn realfft::RealToComplex<f32> + Send + Sync>>,
    fft_input_buf: Option<Vec<f32>>,
    fft_output_buf: Option<Vec<rustfft::num_complex::Complex<f32>>>,
    fft_scratch: Option<Vec<rustfft::num_complex::Complex<f32>>>,
    fft_interleaved: Option<Vec<f32>>,
    hann_window: Option<Vec<f32>>,
    fft_block_size: Option<usize>,
}

fn make_hann_window(n: usize) -> Vec<f32> {
    let n_f = n as f32;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * (i as f32) / n_f).cos()))
        .collect()
}

/// Parallel FFT computation for a batch of audio frames
/// Computes multiple spectra in parallel, each worker gets its own buffers
fn parallel_ffts_interleaved(
    plan: std::sync::Arc<dyn realfft::RealToComplex<f32> + Send + Sync>,
    hann: std::sync::Arc<[f32]>,
    mono: &[f32],
    starts: &[usize],
    block_size: usize,
) -> Vec<Vec<f32>> {
    use rayon::prelude::*;

    starts
        .par_iter()
        .map_init(
            || {
                // Per-thread buffers - each worker allocates once
                let mut input = plan.make_input_vec();
                let mut output = plan.make_output_vec();
                let mut scratch = plan.make_scratch_vec();
                let interleaved_len = output.len() * 2;
                let inter = vec![0.0f32; interleaved_len];
                (input, output, scratch, inter)
            },
            |(input, output, scratch, inter), &start| {
                // Tight copy + window; zero-pad tail if short
                let end = (start + block_size).min(mono.len());
                let got = end - start;

                input[..got].copy_from_slice(&mono[start..end]);
                if got < block_size {
                    input[got..].fill(0.0);
                }

                // Apply Hann window in-place
                apply_hann_inplace(&mut input[..block_size], &hann[..block_size]);

                // Perform FFT
                let _ = plan.process_with_scratch(input, output, scratch);

                // Deinterleave complex -> [re0,im0,re1,im1,...]
                for (k, c) in output.iter().enumerate() {
                    let i = k << 1; // k * 2
                    inter[i] = c.re;
                    inter[i + 1] = c.im;
                }

                std::mem::take(inter) // moves the Vec out; next call allocates a fresh one
            },
        )
        .collect()
}

#[inline]
fn apply_hann_inplace(x: &mut [f32], w: &[f32]) {
    for (xi, wi) in x.iter_mut().zip(w.iter()) {
        *xi *= *wi;
    }
}

#[derive(Debug, Clone)]
struct SimpleFeature {
    has_timestamp: bool,
    sec: i32,
    nsec: i32,
    values: Vec<f32>,
    label: String,
}

impl PyVampPlugin {
    fn ensure_fft_cache(&mut self, block_size: usize) {
        if self.fft_block_size == Some(block_size) {
            return;
        }

        use realfft::RealFftPlanner;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(block_size);

        self.fft_input_buf = Some(fft.make_input_vec());
        self.fft_output_buf = Some(fft.make_output_vec());
        self.fft_scratch = Some(fft.make_scratch_vec());
        self.fft_interleaved = Some(vec![0.0; fft.make_output_vec().len() * 2]);
        self.hann_window = Some(make_hann_window(block_size));
        self.fft_plan = Some(fft);
        self.fft_block_size = Some(block_size);
    }

    fn process_audio_full_inner(
        &mut self,
        audio_samples: Vec<f32>,
        sample_rate: f32,
        channels: usize,
        output_index: Option<usize>,
    ) -> PyResult<Vec<SimpleFeature>> {
        // Ensure plugin is initialized
        if !self.initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Plugin not initialized",
            ));
        }

        // Ensure plugin descriptor and handle are valid
        if self.plugin.deref_mut().descriptor.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Plugin descriptor is null",
            ));
        }
        if self.plugin.deref_mut().handle.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Plugin handle is null",
            ));
        }

        let block_size = self.block_size.unwrap();
        let step_size = self.step_size.unwrap();
        let input_domain = self.plugin.deref_mut().get_input_domain();

        log::debug!("Processing audio: {} samples, {} channels, block_size: {}, step_size: {}, input_domain: {:?}",
                   audio_samples.len(), channels, block_size, step_size, input_domain);

        let mut all_features = Vec::new();

        // Handle frequency domain with parallel FFT batching
        if input_domain == crate::InputDomain::FrequencyDomain {
            self.ensure_fft_cache(block_size);

            // Get shared FFT plan and window with defensive checks
            let plan = self.fft_plan.as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("FFT plan not initialized"))?
                .clone();
            let hann: std::sync::Arc<[f32]> = self.hann_window.as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("Hann window not initialized"))?
                .clone()
                .into();

            // Convert audio to mono buffer first
            let mono_buf = if channels == 1 {
                audio_samples
            } else if channels == 2 {
                // Stereo to mono: mix L+R channels
                let mut mono = Vec::with_capacity(audio_samples.len() / 2);
                for chunk in audio_samples.chunks_exact(2) {
                    mono.push((chunk[0] + chunk[1]) / 2.0);
                }
                mono
            } else {
                // Multi-channel: just take first channel
                let mut mono = Vec::with_capacity(audio_samples.len() / channels);
                for chunk in audio_samples.chunks_exact(channels) {
                    mono.push(chunk[0]);
                }
                mono
            };

            // Precompute all step starts with proper overlap pattern
            let mut starts = Vec::new();
            let mut s = 0usize;
            while s + 1 < mono_buf.len() {  // allow zero-pad tail
                starts.push(s);
                if s + step_size >= mono_buf.len() {  // we'll zero-pad inside FFT
                    break;
                }
                s += step_size;
            }
            // Ensure at least one window
            if starts.is_empty() { 
                starts.push(0); 
            }

            // Batch size for controlling memory usage
            let batch = std::cmp::max(16, rayon::current_num_threads());

            // Process in batches: FFTs in parallel, plugin sequentially
            let mut idx = 0usize;
            while idx < starts.len() {
                let end = (idx + batch).min(starts.len());
                let batch_starts = &starts[idx..end];

                // Keep GIL held to prevent memory issues with plugin
                log::debug!("Processing FFT batch {}-{}", idx, end-1);
                let spectra = parallel_ffts_interleaved(
                    plan.clone(),
                    hann.clone(),
                    &mono_buf,
                    batch_starts,
                    block_size,
                );
                log::debug!("FFT batch completed, {} spectra generated", spectra.len());

                // Now sequentially call plugin.process(...) per spectrum
                for (j, inter) in spectra.into_iter().enumerate() {
                    // Calculate timestamp for this frame
                    let start_samples = batch_starts[j] as i64;
                    let ts_sec = (start_samples as f64 / sample_rate as f64) as i32;
                    let ts_nsec = (((start_samples as f64 / sample_rate as f64) % 1.0)
                        * 1_000_000_000.0) as i32;

                    let buffer_refs: Vec<&[f32]> = vec![inter.as_slice()];

                    log::debug!("Processing FFT frame {} with timestamp {}.{:09}", j, ts_sec, ts_nsec);
                    if let Some(features_ptr) = self.plugin.deref_mut().process(&buffer_refs, ts_sec, ts_nsec) {
                        log::debug!("Got features_ptr from plugin.process for frame {}", j);
                        unsafe {
                            let target_output_index = output_index.unwrap_or(0);

                            // Get output count and validate index to prevent segfault
                            let output_count = if let Some(get_output_count) = (*self.plugin.deref_mut().descriptor).getOutputCount {
                                let count = get_output_count(self.plugin.deref_mut().handle) as usize;
                                log::debug!("Plugin has {} outputs", count);
                                count
                            } else {
                                log::debug!("No getOutputCount function, defaulting to 1 output");
                                1 // Default to 1 output if function not available
                            };

                            if target_output_index >= output_count {
                                log::error!("Output index {} out of bounds (max {})", target_output_index, output_count - 1);
                                self.plugin.deref_mut().release_feature_set(features_ptr);
                                continue;
                            }

                            log::debug!("About to access features at output index {}", target_output_index);
                            let features = &*features_ptr.add(target_output_index);
                            log::debug!("Successfully accessed features pointer, featureCount: {}", features.featureCount);
                            if features.featureCount > 0 && !features.features.is_null() {
                                log::debug!("Features are valid, processing {} features", features.featureCount);
                                for i in 0..features.featureCount {
                                    log::debug!("Accessing feature {}/{}", i+1, features.featureCount);
                                    let feature = &(*features.features.add(i as usize)).v1;
                                    log::debug!("Feature {} accessed successfully", i);
                                    let values =
                                        if feature.valueCount > 0 && !feature.values.is_null() {
                                            std::slice::from_raw_parts(
                                                feature.values,
                                                feature.valueCount as usize,
                                            )
                                            .to_vec()
                                        } else {
                                            Vec::new()
                                        };
                                    let label = if !feature.label.is_null() {
                                        std::ffi::CStr::from_ptr(feature.label)
                                            .to_string_lossy()
                                            .into_owned()
                                    } else {
                                        String::new()
                                    };

                                    all_features.push(SimpleFeature {
                                        has_timestamp: feature.hasTimestamp != 0,
                                        sec: feature.sec,
                                        nsec: feature.nsec,
                                        values,
                                        label,
                                    });
                                }
                            }
                            log::debug!("About to release feature set for frame {}", j);
                            self.plugin.deref_mut().release_feature_set(features_ptr);
                            log::debug!("Feature set released successfully for frame {}", j);
                        }
                    } else {
                        log::debug!("No features returned for frame {}", j);
                    }
                }

                idx = end;
            }
        } else {
            // Time domain processing - keep existing frame-by-frame approach
            let processing_channels = 1; // Always force mono
            let mut timestamp_samples = 0i64;
            let mut sample_index = 0;
            let mut current_step = 0;

            // Calculate overlap size
            let overlap_size = block_size - step_size;

            // Pre-allocate audio buffers
            let mut audio_buffer: Vec<Vec<f32>> = vec![vec![0.0; block_size]; processing_channels];

            let mut final_steps_remaining = if block_size > step_size {
                (block_size / step_size).max(1)
            } else {
                1
            };

            // Main processing loop for time domain
            while final_steps_remaining > 0 {
                let frames_to_read = if (block_size == step_size) || (current_step == 0) {
                    block_size
                } else {
                    // Move existing data down
                    for ch in 0..processing_channels {
                        audio_buffer[ch].copy_within(step_size..block_size, 0);
                    }
                    step_size
                };

                let mut frames_read = 0;
                let start_pos = if (block_size == step_size) || (current_step == 0) {
                    0
                } else {
                    overlap_size
                };

                // Read frames from input, handling stereo to mono conversion
                for frame in 0..frames_to_read {
                    let frame_start = sample_index + frame * channels;
                    if frame_start >= audio_samples.len() {
                        break;
                    }

                    for ch in 0..processing_channels {
                        let buffer_pos = start_pos + frame;
                        if processing_channels == 1 && channels == 2 {
                            // Stereo to mono: mix L+R channels
                            let left_idx = frame_start;
                            let right_idx = frame_start + 1;
                            audio_buffer[0][buffer_pos] = if right_idx < audio_samples.len() {
                                (audio_samples[left_idx] + audio_samples[right_idx]) / 2.0
                            } else {
                                audio_samples[left_idx]
                            };
                        } else {
                            // Direct copy
                            let idx = frame_start + ch;
                            audio_buffer[ch][buffer_pos] = if idx < audio_samples.len() {
                                audio_samples[idx]
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

                // Use audio buffers directly for time domain
                let buffer_refs: Vec<&[f32]> = audio_buffer.iter().map(|v| v.as_slice()).collect();

                // Calculate timestamp
                let timestamp_sec = (timestamp_samples as f64 / sample_rate as f64) as i32;
                let timestamp_nsec = (((timestamp_samples as f64 / sample_rate as f64) % 1.0)
                    * 1_000_000_000.0) as i32;

                // Process this chunk
                if let Some(features_ptr) =
                    self.plugin
                        .process(&buffer_refs, timestamp_sec, timestamp_nsec)
                {
                    unsafe {
                        let target_output_index = output_index.unwrap_or(0);

                        // Get output count and validate index to prevent segfault
                        let output_count = if let Some(get_output_count) = (*self.plugin.deref_mut().descriptor).getOutputCount {
                            get_output_count(self.plugin.deref_mut().handle) as usize
                        } else {
                            1 // Default to 1 output if function not available
                        };

                        if target_output_index >= output_count {
                            log::error!("Output index {} out of bounds (max {})", target_output_index, output_count - 1);
                            self.plugin.deref_mut().release_feature_set(features_ptr);
                            // Don't continue in while loop, just skip this iteration
                        } else {
                            let features = &*features_ptr.add(target_output_index);

                            if features.featureCount > 0 && !features.features.is_null() {
                            for i in 0..features.featureCount {
                                let feature = &(*features.features.add(i as usize)).v1;

                                // Extract values
                                let values = if feature.valueCount > 0 && !feature.values.is_null()
                                {
                                    std::slice::from_raw_parts(
                                        feature.values,
                                        feature.valueCount as usize,
                                    )
                                    .to_vec()
                                } else {
                                    Vec::new()
                                };

                                // Extract label
                                let label = if !feature.label.is_null() {
                                    std::ffi::CStr::from_ptr(feature.label)
                                        .to_string_lossy()
                                        .into_owned()
                                } else {
                                    String::new()
                                };

                                all_features.push(SimpleFeature {
                                    has_timestamp: feature.hasTimestamp != 0,
                                    sec: feature.sec,
                                    nsec: feature.nsec,
                                    values,
                                    label,
                                });
                            }
                        }

                        self.plugin.deref_mut().release_feature_set(features_ptr);
                        } // End of else block for bounds check
                    }
                }

                // Update sample position and step counter
                sample_index += frames_to_read * channels;
                current_step += 1;

                // Update timestamp for next iteration
                timestamp_samples += step_size as i64;
            }
        }

        // Get remaining features
        log::debug!("About to get remaining features from plugin");
        if let Some(remaining_ptr) = self.plugin.deref_mut().get_remaining_features() {
            log::debug!("Got remaining features pointer");
            unsafe {
                let target_output_index = output_index.unwrap_or(0);

                // Get output count and validate index to prevent segfault
                let output_count = if let Some(get_output_count) = (*self.plugin.deref_mut().descriptor).getOutputCount {
                    get_output_count(self.plugin.deref_mut().handle) as usize
                } else {
                    1 // Default to 1 output if function not available
                };

                if target_output_index >= output_count {
                    log::error!("Output index {} out of bounds (max {})", target_output_index, output_count - 1);
                    self.plugin.deref_mut().release_feature_set(remaining_ptr);
                    return Ok(all_features);
                }

                let remaining = &*remaining_ptr.add(target_output_index);

                if remaining.featureCount > 0 && !remaining.features.is_null() {
                    for i in 0..remaining.featureCount {
                        let feature = &(*remaining.features.add(i as usize)).v1;

                        let values = if feature.valueCount > 0 && !feature.values.is_null() {
                            std::slice::from_raw_parts(feature.values, feature.valueCount as usize)
                                .to_vec()
                        } else {
                            Vec::new()
                        };

                        let label = if !feature.label.is_null() {
                            std::ffi::CStr::from_ptr(feature.label)
                                .to_string_lossy()
                                .into_owned()
                        } else {
                            String::new()
                        };

                        all_features.push(SimpleFeature {
                            has_timestamp: feature.hasTimestamp != 0,
                            sec: feature.sec,
                            nsec: feature.nsec,
                            values,
                            label,
                        });
                    }
                }

                log::debug!("About to release remaining feature set");
                self.plugin.deref_mut().release_feature_set(remaining_ptr);
                log::debug!("Remaining feature set released successfully");
            }
        } else {
            log::debug!("No remaining features returned");
        }

        log::debug!("Total features collected: {}", all_features.len());
        Ok(all_features)
    }

    fn convert_features_to_python(
        &self,
        py: Python,
        features: &crate::VampFeatureList,
    ) -> PyResult<PyObject> {
        // Convert features to Python objects
        let result = PyDict::new_bound(py);
        result.set_item("feature_count", features.featureCount)?;

        // Create a list of features
        let feature_list = PyList::empty_bound(py);

        if features.featureCount > 0 && !features.features.is_null() {
            unsafe {
                let features_slice =
                    std::slice::from_raw_parts(features.features, features.featureCount as usize);

                for feature in features_slice {
                    let feature_dict = PyDict::new_bound(py);
                    let vamp_feature = &feature.v1;
                    feature_dict.set_item("has_timestamp", vamp_feature.hasTimestamp != 0)?;

                    if vamp_feature.hasTimestamp != 0 {
                        feature_dict.set_item("sec", vamp_feature.sec)?;
                        feature_dict.set_item("nsec", vamp_feature.nsec)?;
                    }

                    // Convert values to Python list
                    if vamp_feature.valueCount > 0 && !vamp_feature.values.is_null() {
                        let values_slice = std::slice::from_raw_parts(
                            vamp_feature.values,
                            vamp_feature.valueCount as usize,
                        );
                        let values_list: Vec<f32> = values_slice.to_vec();
                        feature_dict.set_item("values", values_list)?;
                    } else {
                        let empty_list = PyList::empty_bound(py);
                        feature_dict.set_item("values", empty_list)?;
                    }

                    // Convert label if present
                    if !vamp_feature.label.is_null() {
                        let label_cstr = std::ffi::CStr::from_ptr(vamp_feature.label);
                        if let Ok(label_str) = label_cstr.to_str() {
                            feature_dict.set_item("label", label_str)?;
                        }
                    }

                    feature_list.append(feature_dict)?;
                }
            }
        }

        result.set_item("features", feature_list)?;
        Ok(result.into())
    }
}

#[pymethods]
impl PyVampPlugin {
    fn get_input_domain(&self) -> PyInputDomain {
        match self.plugin.deref().get_input_domain() {
            InputDomain::TimeDomain => PyInputDomain::TimeDomain,
            InputDomain::FrequencyDomain => PyInputDomain::FrequencyDomain,
        }
    }

    fn get_preferred_block_size(&self) -> u32 {
        self.plugin.deref().get_preferred_block_size()
    }

    fn get_preferred_step_size(&self) -> u32 {
        self.plugin.deref().get_preferred_step_size()
    }

    fn initialise(&mut self, channels: u32, step_size: u32, block_size: u32) -> bool {
        self.plugin.deref_mut().initialise(channels, step_size, block_size)
    }

    fn process(
        &mut self,
        input_buffers: Vec<Vec<f32>>,
        sec: i32,
        nsec: i32,
    ) -> PyResult<Option<PyObject>> {
        self.process_with_output(input_buffers, sec, nsec, Some(0))
    }

    #[pyo3(signature = (input_buffers, sec, nsec, output_index=None))]
    fn process_with_output(
        &mut self,
        input_buffers: Vec<Vec<f32>>,
        sec: i32,
        nsec: i32,
        output_index: Option<usize>,
    ) -> PyResult<Option<PyObject>> {
        Python::with_gil(|py| {
            // Convert Vec<Vec<f32>> to the format expected by the plugin
            let buffer_refs: Vec<&[f32]> = input_buffers.iter().map(|v| v.as_slice()).collect();

            match self.plugin.deref_mut().process(&buffer_refs, sec, nsec) {
                Some(features_ptr) => {
                    unsafe {
                        // If output_index is specified, get features from that output
                        // Otherwise return all outputs
                        if let Some(idx) = output_index {
                            let features = &*features_ptr.add(idx);
                            let result = self.convert_features_to_python(py, features)?;

                            // Release the feature set to avoid memory leaks
                            self.plugin.deref_mut().release_feature_set(features_ptr);

                            Ok(Some(result.into()))
                        } else {
                            // For safety, just return the first output when no specific output requested
                            // TODO: Implement proper output descriptor querying to safely access all outputs
                            let features = &*features_ptr;
                            let result = self.convert_features_to_python(py, features)?;

                            // Release the feature set to avoid memory leaks
                            self.plugin.deref_mut().release_feature_set(features_ptr);

                            Ok(Some(result.into()))
                        }
                    }
                }
                None => Ok(None),
            }
        })
    }

    fn get_remaining_features(&mut self) -> PyResult<Option<PyObject>> {
        self.get_remaining_features_with_output(Some(0))
    }

    #[pyo3(signature = (output_index=None))]
    fn get_remaining_features_with_output(
        &mut self,
        output_index: Option<usize>,
    ) -> PyResult<Option<PyObject>> {
        Python::with_gil(|py| {
            match self.plugin.deref_mut().get_remaining_features() {
                Some(features_ptr) => {
                    unsafe {
                        if let Some(idx) = output_index {
                            let features = &*features_ptr.add(idx);
                            let result = self.convert_features_to_python(py, features)?;

                            // Release the feature set
                            self.plugin.deref_mut().release_feature_set(features_ptr);

                            Ok(Some(result))
                        } else {
                            // For safety, just return the first output when no specific output requested
                            // TODO: Implement proper output descriptor querying to safely access all outputs
                            let features = &*features_ptr;
                            let result = self.convert_features_to_python(py, features)?;

                            // Release the feature set
                            self.plugin.deref_mut().release_feature_set(features_ptr);

                            Ok(Some(result))
                        }
                    }
                }
                None => Ok(None),
            }
        })
    }

    fn reset(&mut self) {
        self.plugin.deref_mut().reset();
        self.initialized = false;
        self.block_size = None;
        self.step_size = None;
        // Clear FFT cache
        self.fft_plan = None;
        self.fft_input_buf = None;
        self.fft_output_buf = None;
        self.fft_scratch = None;
        self.fft_interleaved = None;
        self.hann_window = None;
        self.fft_block_size = None;
    }

    #[pyo3(signature = (sample_rate, channels=1))]
    fn initialize(&mut self, sample_rate: f32, channels: Option<usize>) -> PyResult<()> {
        if self.initialized {
            return Ok(()); // Already initialized
        }

        let processing_channels = channels.unwrap_or(1);

        // Get preferred block and step sizes from the plugin
        let preferred_block_size = unsafe {
            if !self.plugin.deref_mut().descriptor.is_null() && 
               (*self.plugin.deref_mut().descriptor).getPreferredBlockSize.is_some() {
                if let Some(get_preferred_block) = (*self.plugin.deref_mut().descriptor).getPreferredBlockSize {
                    get_preferred_block(self.plugin.deref_mut().handle)
                } else {
                    1024
                }
            } else {
                1024
            }
        };

        let preferred_step_size = unsafe {
            if !self.plugin.deref_mut().descriptor.is_null() && 
               (*self.plugin.deref_mut().descriptor).getPreferredStepSize.is_some() {
                if let Some(get_preferred_step) = (*self.plugin.deref_mut().descriptor).getPreferredStepSize {
                    get_preferred_step(self.plugin.deref_mut().handle)
                } else {
                    preferred_block_size / 2
                }
            } else {
                preferred_block_size / 2
            }
        };

        let block_size = if preferred_block_size > 0 {
            preferred_block_size as usize
        } else {
            1024
        };

        let input_domain = self.plugin.deref_mut().get_input_domain();

        let step_size = if preferred_step_size > 0 {
            preferred_step_size as usize
        } else {
            if input_domain == crate::InputDomain::FrequencyDomain {
                block_size / 2 // 50% overlap for frequency domain
            } else {
                block_size // No overlap for time domain
            }
        };

        if !self.plugin.deref_mut().initialise(
            processing_channels as u32,
            step_size as u32,
            block_size as u32,
        ) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to initialize plugin",
            ));
        }

        self.initialized = true;
        self.block_size = Some(block_size);
        self.step_size = Some(step_size);
        Ok(())
    }

    #[pyo3(signature = (audio_samples, sample_rate, channels, output_index=None))]
    fn process_audio_full(
        &mut self,
        audio_samples: Vec<f32>,
        sample_rate: f32,
        channels: usize,
        output_index: Option<usize>,
    ) -> PyResult<Option<PyObject>> {
        // Ensure plugin is initialized first (auto-initialize if needed)
        if !self.initialized {
            self.initialize(sample_rate, Some(1))?; // Force mono
        }

        // Process audio - we'll release GIL only for the FFT heavy work
        let result = self
            .process_audio_full_inner(audio_samples, sample_rate, channels, output_index)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // Reacquire GIL to convert results to Python
        Python::with_gil(|py| {
            let all_features = PyList::empty_bound(py);

            // Convert SimpleFeature structs to Python dicts
            for feature in result {
                let feature_dict = PyDict::new_bound(py);
                feature_dict.set_item("has_timestamp", feature.has_timestamp)?;

                if feature.has_timestamp {
                    feature_dict.set_item("sec", feature.sec)?;
                    feature_dict.set_item("nsec", feature.nsec)?;
                }

                feature_dict.set_item("values", feature.values)?;

                if !feature.label.is_empty() {
                    feature_dict.set_item("label", feature.label)?;
                }

                all_features.append(feature_dict)?;
            }

            Ok(Some(all_features.into()))
        })
    }

    #[cfg(feature = "python")]
    #[pyo3(signature = (audio, sample_rate, channels=None, output_index=None))]
    fn process_audio_nd(
        &mut self,
        py: Python<'_>,
        audio: Bound<'_, PyAny>,
        sample_rate: f32,
        channels: Option<usize>,
        output_index: Option<usize>,
    ) -> PyResult<Option<PyObject>> {
        // Accept either 1D (mono) or 2D (frames, channels) float32 arrays
        if let Ok(a1) = audio.extract::<PyReadonlyArray1<f32>>() {
            let mono: &[f32] = a1.as_slice()?; // zero-copy view while GIL held
            self.process_audio_internal(py, mono.to_vec(), sample_rate, 1, output_index)
        } else if let Ok(a2) = audio.extract::<PyReadonlyArray2<f32>>() {
            let a2 = a2.as_array(); // ndarray view
            let (frames, ch) = (a2.shape()[0], a2.shape()[1]);
            // If your plugin expects interleaved, do one tight pack here:
            let mut interleaved = Vec::<f32>::with_capacity(frames * ch);
            // contiguous row-major assumed; if not, copy via .to_owned()
            for i in 0..frames {
                for c in 0..ch {
                    interleaved.push(a2[[i, c]]);
                }
            }
            self.process_audio_internal(py, interleaved, sample_rate, ch, output_index)
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "audio must be float32 NumPy array of shape (T,) or (T, C)",
            ))
        }
    }

    #[pyo3(signature = (interleaved, sample_rate, channels, output_index=None))]
    fn process_audio_internal(
        &mut self,
        py: Python<'_>,
        interleaved: Vec<f32>,
        sample_rate: f32,
        channels: usize,
        output_index: Option<usize>,
    ) -> PyResult<Option<PyObject>> {
        // Ensure plugin is initialized first (auto-initialize if needed)
        if !self.initialized {
            self.initialize(sample_rate, Some(1))?; // Force mono
        }

        // Process audio while holding GIL to prevent memory issues
        log::debug!("About to process audio with {} samples", interleaved.len());
        let result = self
            .process_audio_full_inner(interleaved, sample_rate, channels, output_index)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        log::debug!("Audio processing completed, converting {} features to Python", result.len());

        // Convert results to Python objects
        let all_features = PyList::empty_bound(py);

        // Convert SimpleFeature structs to Python dicts
        for (i, feature) in result.iter().enumerate() {
            log::debug!("Converting feature {} to Python dict", i);
            let feature_dict = PyDict::new_bound(py);
            feature_dict.set_item("has_timestamp", feature.has_timestamp)?;

            if feature.has_timestamp {
                feature_dict.set_item("sec", feature.sec)?;
                feature_dict.set_item("nsec", feature.nsec)?;
            }

            // Create Python list for values to ensure proper ownership
            let py_values = PyList::empty_bound(py);
            for val in &feature.values {
                py_values.append(*val)?;
            }
            feature_dict.set_item("values", py_values)?;

            if !feature.label.is_empty() {
                // Create Python string to ensure proper ownership
                let py_label = pyo3::types::PyString::new_bound(py, &feature.label);
                feature_dict.set_item("label", py_label)?;
            }

            all_features.append(feature_dict)?;
            log::debug!("Feature {} converted successfully", i);
        }

        log::debug!("All {} features converted to Python, returning result", result.len());
        let py_obj = all_features.into();
        log::debug!("Converted PyList to PyObject");
        Ok(Some(py_obj))
    }

    fn get_output_descriptors(&mut self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let outputs = PyList::empty_bound(py);

            unsafe {
                let desc = &*self.plugin.deref().descriptor;

                // First get the output count using getOutputCount
                if let Some(get_output_count_fn) = desc.getOutputCount {
                    let output_count = get_output_count_fn(self.plugin.deref_mut().handle);

                    if output_count > 0 {
                        // Then get each output descriptor using getOutputDescriptor
                        if let Some(get_output_descriptor_fn) = desc.getOutputDescriptor {
                            for i in 0..output_count {
                                let output_desc = get_output_descriptor_fn(self.plugin.deref_mut().handle, i);
                                if !output_desc.is_null() {
                                    let output_ref = &*output_desc;
                                    let output_dict = PyDict::new_bound(py);

                                    output_dict.set_item("index", i)?;

                                    // Get identifier
                                    if !output_ref.identifier.is_null() {
                                        let id = std::ffi::CStr::from_ptr(output_ref.identifier)
                                            .to_string_lossy();
                                        output_dict.set_item("identifier", id.as_ref())?;
                                    } else {
                                        output_dict
                                            .set_item("identifier", format!("output_{}", i))?;
                                    }

                                    // Get name
                                    if !output_ref.name.is_null() {
                                        let name = std::ffi::CStr::from_ptr(output_ref.name)
                                            .to_string_lossy();
                                        output_dict.set_item("name", name.as_ref())?;
                                    } else {
                                        output_dict.set_item("name", format!("Output {}", i))?;
                                    }

                                    // Get description
                                    if !output_ref.description.is_null() {
                                        let desc_str =
                                            std::ffi::CStr::from_ptr(output_ref.description)
                                                .to_string_lossy();
                                        output_dict.set_item("description", desc_str.as_ref())?;
                                    } else {
                                        output_dict.set_item("description", "")?;
                                    }

                                    // Get unit
                                    if !output_ref.unit.is_null() {
                                        let unit = std::ffi::CStr::from_ptr(output_ref.unit)
                                            .to_string_lossy();
                                        output_dict.set_item("unit", unit.as_ref())?;
                                    } else {
                                        output_dict.set_item("unit", "")?;
                                    }

                                    // Binary count information
                                    output_dict.set_item(
                                        "has_fixed_bin_count",
                                        output_ref.hasFixedBinCount != 0,
                                    )?;
                                    output_dict.set_item("bin_count", output_ref.binCount)?;

                                    // Bin names (if available)
                                    if output_ref.hasFixedBinCount != 0
                                        && !output_ref.binNames.is_null()
                                    {
                                        let bin_names = PyList::empty_bound(py);
                                        for bin_idx in 0..output_ref.binCount {
                                            let bin_name_ptr =
                                                *output_ref.binNames.add(bin_idx as usize);
                                            if !bin_name_ptr.is_null() {
                                                let bin_name =
                                                    std::ffi::CStr::from_ptr(bin_name_ptr)
                                                        .to_string_lossy();
                                                bin_names.append(bin_name.as_ref())?;
                                            } else {
                                                bin_names.append(format!("Bin {}", bin_idx))?;
                                            }
                                        }
                                        output_dict.set_item("bin_names", bin_names)?;
                                    } else {
                                        output_dict
                                            .set_item("bin_names", PyList::empty_bound(py))?;
                                    }

                                    // Extents information
                                    output_dict.set_item(
                                        "has_known_extents",
                                        output_ref.hasKnownExtents != 0,
                                    )?;
                                    if output_ref.hasKnownExtents != 0 {
                                        output_dict.set_item("min_value", output_ref.minValue)?;
                                        output_dict.set_item("max_value", output_ref.maxValue)?;
                                    } else {
                                        output_dict.set_item("min_value", py.None())?;
                                        output_dict.set_item("max_value", py.None())?;
                                    }

                                    // Quantization information
                                    output_dict
                                        .set_item("is_quantized", output_ref.isQuantized != 0)?;
                                    if output_ref.isQuantized != 0 {
                                        output_dict
                                            .set_item("quantize_step", output_ref.quantizeStep)?;
                                    } else {
                                        output_dict.set_item("quantize_step", py.None())?;
                                    }

                                    // Sample type and rate
                                    let sample_type_str = match output_ref.sampleType {
                                        0 => "OneSamplePerStep",   // vampOneSamplePerStep
                                        1 => "FixedSampleRate",    // vampFixedSampleRate
                                        2 => "VariableSampleRate", // vampVariableSampleRate
                                        _ => "Unknown",
                                    };
                                    output_dict.set_item("sample_type", sample_type_str)?;
                                    output_dict.set_item("sample_rate", output_ref.sampleRate)?;

                                    // Duration support (API version 2+)
                                    if desc.vampApiVersion >= 2 {
                                        output_dict.set_item(
                                            "has_duration",
                                            output_ref.hasDuration != 0,
                                        )?;
                                    } else {
                                        output_dict.set_item("has_duration", false)?;
                                    }

                                    outputs.append(output_dict)?;

                                    // Release the output descriptor
                                    if let Some(release_fn) = desc.releaseOutputDescriptor {
                                        release_fn(output_desc);
                                    }
                                }
                            }
                        }
                    }
                }

                // Fallback if no outputs found
                if outputs.len() == 0 {
                    let output_dict = PyDict::new_bound(py);
                    output_dict.set_item("index", 0)?;
                    output_dict.set_item("identifier", "output")?;
                    output_dict.set_item("name", "Default Output")?;
                    output_dict.set_item("description", "")?;
                    output_dict.set_item("unit", "")?;
                    output_dict.set_item("has_fixed_bin_count", false)?;
                    output_dict.set_item("bin_count", 1u32)?;
                    output_dict.set_item("bin_names", PyList::empty_bound(py))?;
                    output_dict.set_item("has_known_extents", false)?;
                    output_dict.set_item("min_value", py.None())?;
                    output_dict.set_item("max_value", py.None())?;
                    output_dict.set_item("is_quantized", false)?;
                    output_dict.set_item("quantize_step", py.None())?;
                    output_dict.set_item("sample_type", "OneSamplePerStep")?;
                    output_dict.set_item("sample_rate", 0.0f32)?;
                    output_dict.set_item("has_duration", false)?;
                    outputs.append(output_dict)?;
                }
            }

            Ok(outputs.into())
        })
    }

    fn get_parameter_descriptors(&mut self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let parameters = PyList::empty_bound(py);

            unsafe {
                let desc = &*self.plugin.deref().descriptor;

                // Get parameter descriptors
                if desc.parameterCount > 0 && !desc.parameters.is_null() {
                    for i in 0..desc.parameterCount {
                        let param_ptr = *desc.parameters.add(i as usize);
                        if !param_ptr.is_null() {
                            let param_ref = &*param_ptr;
                            let param_dict = PyDict::new_bound(py);

                            param_dict.set_item("index", i)?;

                            // Get identifier
                            if !param_ref.identifier.is_null() {
                                let id = std::ffi::CStr::from_ptr(param_ref.identifier)
                                    .to_string_lossy();
                                param_dict.set_item("identifier", id.as_ref())?;
                            } else {
                                param_dict.set_item("identifier", format!("param_{}", i))?;
                            }

                            // Get name
                            if !param_ref.name.is_null() {
                                let name =
                                    std::ffi::CStr::from_ptr(param_ref.name).to_string_lossy();
                                param_dict.set_item("name", name.as_ref())?;
                            } else {
                                param_dict.set_item("name", format!("Parameter {}", i))?;
                            }

                            // Get description
                            if !param_ref.description.is_null() {
                                let desc_str = std::ffi::CStr::from_ptr(param_ref.description)
                                    .to_string_lossy();
                                param_dict.set_item("description", desc_str.as_ref())?;
                            } else {
                                param_dict.set_item("description", "")?;
                            }

                            // Get unit
                            if !param_ref.unit.is_null() {
                                let unit =
                                    std::ffi::CStr::from_ptr(param_ref.unit).to_string_lossy();
                                param_dict.set_item("unit", unit.as_ref())?;
                            } else {
                                param_dict.set_item("unit", "")?;
                            }

                            // Value range
                            param_dict.set_item("min_value", param_ref.minValue)?;
                            param_dict.set_item("max_value", param_ref.maxValue)?;
                            param_dict.set_item("default_value", param_ref.defaultValue)?;

                            // Quantization
                            param_dict.set_item("is_quantized", param_ref.isQuantized != 0)?;
                            if param_ref.isQuantized != 0 {
                                param_dict.set_item("quantize_step", param_ref.quantizeStep)?;

                                // Value names for quantized parameters
                                if !param_ref.valueNames.is_null() {
                                    let value_names = PyList::empty_bound(py);
                                    let mut idx = 0;
                                    loop {
                                        let value_name_ptr = *param_ref.valueNames.add(idx);
                                        if value_name_ptr.is_null() {
                                            break;
                                        }
                                        let value_name = std::ffi::CStr::from_ptr(value_name_ptr)
                                            .to_string_lossy();
                                        value_names.append(value_name.as_ref())?;
                                        idx += 1;
                                    }
                                    param_dict.set_item("value_names", value_names)?;
                                } else {
                                    param_dict.set_item("value_names", PyList::empty_bound(py))?;
                                }
                            } else {
                                param_dict.set_item("quantize_step", py.None())?;
                                param_dict.set_item("value_names", PyList::empty_bound(py))?;
                            }

                            parameters.append(param_dict)?;
                        }
                    }
                }
            }

            Ok(parameters.into())
        })
    }

    fn get_plugin_info(&mut self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let info_dict = PyDict::new_bound(py);

            unsafe {
                let desc = &*self.plugin.deref().descriptor;

                // Get plugin identifier
                if !desc.identifier.is_null() {
                    let identifier = std::ffi::CStr::from_ptr(desc.identifier).to_string_lossy();
                    info_dict.set_item("identifier", identifier.as_ref())?;
                }

                // Get plugin name
                if !desc.name.is_null() {
                    let name = std::ffi::CStr::from_ptr(desc.name).to_string_lossy();
                    info_dict.set_item("name", name.as_ref())?;
                }

                // Get description
                if !desc.description.is_null() {
                    let description = std::ffi::CStr::from_ptr(desc.description).to_string_lossy();
                    info_dict.set_item("description", description.as_ref())?;
                }

                // Get maker
                if !desc.maker.is_null() {
                    let maker = std::ffi::CStr::from_ptr(desc.maker).to_string_lossy();
                    info_dict.set_item("maker", maker.as_ref())?;
                }

                // Get version
                info_dict.set_item("version", desc.pluginVersion)?;

                // Get copyright
                if !desc.copyright.is_null() {
                    let copyright = std::ffi::CStr::from_ptr(desc.copyright).to_string_lossy();
                    info_dict.set_item("copyright", copyright.as_ref())?;
                }

                // Add API version
                info_dict.set_item("api_version", desc.vampApiVersion)?;

                // Add parameter and program counts
                info_dict.set_item("parameter_count", desc.parameterCount)?;
                info_dict.set_item("program_count", desc.programCount)?;

                // Add sample rate
                info_dict.set_item("sample_rate", self.plugin.deref().sample_rate)?;

                // Add input domain
                let input_domain_str = match self.get_input_domain() {
                    PyInputDomain::TimeDomain => "TimeDomain",
                    PyInputDomain::FrequencyDomain => "FrequencyDomain",
                };
                info_dict.set_item("input_domain", input_domain_str)?;

                // Add preferred sizes
                info_dict.set_item("preferred_block_size", self.get_preferred_block_size())?;
                info_dict.set_item("preferred_step_size", self.get_preferred_step_size())?;
            }

            Ok(info_dict.into())
        })
    }

    fn set_parameter(&mut self, parameter: u32, value: f32) -> bool {
        self.plugin.deref_mut().set_parameter(parameter, value)
    }

    fn get_parameter(&self, parameter: u32) -> Option<f32> {
        self.plugin.deref().get_parameter(parameter)
    }

    fn set_parameter_by_name(&mut self, name: &str, value: f32) -> PyResult<bool> {
        Python::with_gil(|_py| {
            unsafe {
                let desc = &*self.plugin.deref().descriptor;

                // Find parameter by identifier
                if desc.parameterCount > 0 && !desc.parameters.is_null() {
                    for i in 0..desc.parameterCount {
                        let param_ptr = *desc.parameters.add(i as usize);
                        if !param_ptr.is_null() {
                            let param_ref = &*param_ptr;
                            if !param_ref.identifier.is_null() {
                                let id = std::ffi::CStr::from_ptr(param_ref.identifier)
                                    .to_string_lossy();
                                if id == name {
                                    return Ok(self.plugin.deref_mut().set_parameter(i, value));
                                }
                            }
                        }
                    }
                }

                Err(PyRuntimeError::new_err(format!(
                    "Parameter '{}' not found",
                    name
                )))
            }
        })
    }

    fn get_parameter_by_name(&mut self, name: &str) -> PyResult<Option<f32>> {
        Python::with_gil(|_py| {
            unsafe {
                let desc = &*self.plugin.deref().descriptor;

                // Find parameter by identifier
                if desc.parameterCount > 0 && !desc.parameters.is_null() {
                    for i in 0..desc.parameterCount {
                        let param_ptr = *desc.parameters.add(i as usize);
                        if !param_ptr.is_null() {
                            let param_ref = &*param_ptr;
                            if !param_ref.identifier.is_null() {
                                let id = std::ffi::CStr::from_ptr(param_ref.identifier)
                                    .to_string_lossy();
                                if id == name {
                                    return Ok(self.plugin.deref().get_parameter(i));
                                }
                            }
                        }
                    }
                }

                Err(PyRuntimeError::new_err(format!(
                    "Parameter '{}' not found",
                    name
                )))
            }
        })
    }

    fn set_parameters(&mut self, parameters: &Bound<PyDict>) -> PyResult<()> {
        for (key, value) in parameters {
            let param_name: String = key.extract()?;
            let param_value: f32 = value.extract()?;
            self.set_parameter_by_name(&param_name, param_value)?;
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("VampPlugin(sample_rate={})", self.plugin.deref().sample_rate)
    }

    fn __del__(&mut self) {
        // During Python shutdown, we need to avoid segfaults from cleanup
        // Since we're using ManuallyDrop, the plugin won't be automatically dropped
        // We intentionally leak the memory to avoid calling cleanup on a potentially
        // invalid function pointer after the library has been unloaded
        // See: https://github.com/PyO3/pyo3/issues/4632

        // Do nothing - ManuallyDrop prevents automatic cleanup
        // This leaks memory but prevents segfaults
    }
}

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyInputDomain {
    TimeDomain,
    FrequencyDomain,
}

#[pymethods]
impl PyInputDomain {
    fn __repr__(&self) -> String {
        match self {
            PyInputDomain::TimeDomain => "InputDomain.TimeDomain".to_string(),
            PyInputDomain::FrequencyDomain => "InputDomain.FrequencyDomain".to_string(),
        }
    }
}

#[pyclass]
pub struct PyVampError {
    message: String,
}

#[pymethods]
impl PyVampError {
    #[new]
    fn new(message: String) -> Self {
        PyVampError { message }
    }

    fn __str__(&self) -> String {
        self.message.clone()
    }

    fn __repr__(&self) -> String {
        format!("VampError('{}')", self.message)
    }
}

impl std::convert::From<PyVampError> for PyErr {
    fn from(_: PyVampError) -> PyErr {
        PyRuntimeError::new_err("VampError")
    }
}

#[pymodule]
fn _vamprust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize env_logger if not already initialized
    // This allows RUST_LOG env var to control debug output
    let _ = env_logger::try_init();
    
    log::debug!("Initializing vamprust Python module");

    m.add_class::<PyVampHost>()?;
    m.add_class::<PyVampLibrary>()?;
    m.add_class::<PyVampPlugin>()?;
    m.add_class::<PyPluginInfo>()?;
    m.add_class::<PyInputDomain>()?;
    m.add_class::<PyVampError>()?;

    // Add aliases for cleaner Python API
    m.add("VampHost", m.getattr("PyVampHost")?)?;
    m.add("VampLibrary", m.getattr("PyVampLibrary")?)?;
    m.add("VampPlugin", m.getattr("PyVampPlugin")?)?;
    m.add("PluginInfo", m.getattr("PyPluginInfo")?)?;
    m.add("InputDomain", m.getattr("PyInputDomain")?)?;
    m.add("VampError", m.getattr("PyVampError")?)?;

    log::debug!("vamprust Python module initialized successfully");
    Ok(())
}
