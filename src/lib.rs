use libc::{c_uint, c_void, dlclose, dlopen, dlsym, RTLD_LAZY};
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::{env, fs};

#[cfg(feature = "python")]
pub mod python;

mod ffi {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use ffi::*;

#[derive(Clone)]
pub struct VampHost {
    pub plugin_paths: Vec<PathBuf>,
}

impl Default for VampHost {
    fn default() -> Self {
        Self::new()
    }
}

impl VampHost {
    pub fn new() -> Self {
        let mut plugin_paths = Vec::new();

        // Add default Vamp plugin paths
        if let Some(vamp_path) = env::var_os("VAMP_PATH") {
            for path in env::split_paths(&vamp_path) {
                plugin_paths.push(path);
            }
        } else {
            // Default search paths
            #[cfg(target_os = "macos")]
            {
                plugin_paths.push(PathBuf::from("/Library/Audio/Plug-Ins/Vamp"));
                if let Some(home) = env::var_os("HOME") {
                    plugin_paths.push(PathBuf::from(home).join("Library/Audio/Plug-Ins/Vamp"));
                }
            }

            #[cfg(target_os = "linux")]
            {
                plugin_paths.push(PathBuf::from("/usr/lib/vamp"));
                plugin_paths.push(PathBuf::from("/usr/local/lib/vamp"));
                if let Some(home) = env::var_os("HOME") {
                    plugin_paths.push(PathBuf::from(home).join(".vamp"));
                }
            }
        }

        VampHost { plugin_paths }
    }

    pub fn find_plugin_libraries(&self) -> Vec<PathBuf> {
        let mut libraries = Vec::new();

        for search_path in &self.plugin_paths {
            if let Ok(entries) = fs::read_dir(search_path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(extension) = path.extension() {
                        #[cfg(target_os = "macos")]
                        let is_plugin = extension == "dylib";
                        #[cfg(target_os = "linux")]
                        let is_plugin = extension == "so";
                        #[cfg(target_os = "windows")]
                        let is_plugin = extension == "dll";

                        if is_plugin {
                            libraries.push(path);
                        }
                    }
                }
            }
        }

        libraries
    }

    pub fn load_library<P: AsRef<Path>>(&self, library_path: P) -> Option<VampLibrary> {
        VampLibrary::load(library_path)
    }
}

pub struct VampLibrary {
    handle: *mut c_void,
    path: PathBuf,
}

type VampGetPluginDescriptorFn = extern "C" fn(c_uint, c_uint) -> *const VampPluginDescriptor;

impl VampLibrary {
    pub fn load<P: AsRef<Path>>(library_path: P) -> Option<Self> {
        let path = library_path.as_ref();
        let path_cstr = CString::new(path.to_str()?).ok()?;

        unsafe {
            let handle = dlopen(path_cstr.as_ptr(), RTLD_LAZY);
            if handle.is_null() {
                return None;
            }

            Some(VampLibrary {
                handle,
                path: path.to_path_buf(),
            })
        }
    }

    fn get_plugin_descriptor_fn(&self) -> Option<VampGetPluginDescriptorFn> {
        unsafe {
            let symbol_name = CString::new("vampGetPluginDescriptor").ok()?;
            let symbol = dlsym(self.handle, symbol_name.as_ptr());
            if symbol.is_null() {
                return None;
            }
            Some(std::mem::transmute::<*mut c_void, VampGetPluginDescriptorFn>(symbol))
        }
    }

    pub fn get_plugin_descriptor(&self, plugin_index: usize) -> Option<&VampPluginDescriptor> {
        let get_descriptor = self.get_plugin_descriptor_fn()?;

        unsafe {
            let desc = get_descriptor(VAMP_API_VERSION, plugin_index as c_uint);
            if desc.is_null() {
                None
            } else {
                Some(&*desc)
            }
        }
    }

    pub fn list_plugins(&self) -> Vec<PluginInfo> {
        let mut plugins = Vec::new();
        let mut index = 0;

        while let Some(desc) = self.get_plugin_descriptor(index) {
            unsafe {
                let identifier = if desc.identifier.is_null() {
                    format!("plugin_{}", index)
                } else {
                    CStr::from_ptr(desc.identifier)
                        .to_string_lossy()
                        .into_owned()
                };

                let name = if desc.name.is_null() {
                    identifier.clone()
                } else {
                    CStr::from_ptr(desc.name).to_string_lossy().into_owned()
                };

                plugins.push(PluginInfo {
                    identifier,
                    name,
                    index,
                    library_path: self.path.clone(),
                });
            }
            index += 1;
        }

        plugins
    }

    pub fn instantiate_plugin(&self, plugin_index: usize, sample_rate: f32) -> Option<VampPlugin> {
        if let Some(desc) = self.get_plugin_descriptor(plugin_index) {
            unsafe {
                if let Some(instantiate) = desc.instantiate {
                    let handle = instantiate(desc, sample_rate);
                    if !handle.is_null() {
                        return Some(VampPlugin {
                            handle,
                            descriptor: desc,
                            sample_rate,
                        });
                    }
                }
            }
        }
        None
    }
}

impl Drop for VampLibrary {
    fn drop(&mut self) {
        unsafe {
            dlclose(self.handle);
        }
    }
}

#[derive(Debug, Clone)]
pub struct PluginInfo {
    pub identifier: String,
    pub name: String,
    pub index: usize,
    pub library_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InputDomain {
    TimeDomain,
    FrequencyDomain,
}

pub struct VampPlugin {
    pub handle: VampPluginHandle,
    pub descriptor: *const VampPluginDescriptor,
    pub sample_rate: f32,
}

impl VampPlugin {
    pub fn get_input_domain(&self) -> InputDomain {
        unsafe {
            match (*self.descriptor).inputDomain {
                0 => InputDomain::TimeDomain,      // vampTimeDomain
                1 => InputDomain::FrequencyDomain, // vampFrequencyDomain
                _ => InputDomain::TimeDomain,      // default
            }
        }
    }

    pub fn get_preferred_block_size(&self) -> u32 {
        unsafe {
            if let Some(get_preferred_block) = (*self.descriptor).getPreferredBlockSize {
                get_preferred_block(self.handle)
            } else {
                1024 // default
            }
        }
    }

    pub fn get_preferred_step_size(&self) -> u32 {
        unsafe {
            if let Some(get_preferred_step) = (*self.descriptor).getPreferredStepSize {
                get_preferred_step(self.handle)
            } else {
                0 // 0 means use default based on domain
            }
        }
    }

    pub fn initialise(&mut self, channels: u32, step_size: u32, block_size: u32) -> bool {
        unsafe {
            if let Some(initialise) = (*self.descriptor).initialise {
                initialise(self.handle, channels, step_size, block_size) != 0
            } else {
                false
            }
        }
    }

    pub fn process(
        &mut self,
        input_buffers: &[&[f32]],
        sec: i32,
        nsec: i32,
    ) -> Option<*mut VampFeatureList> {
        unsafe {
            if let Some(process) = (*self.descriptor).process {
                let result = process(
                    self.handle,
                    input_buffers.as_ptr() as *const *const f32,
                    sec,
                    nsec,
                );
                if result.is_null() {
                    None
                } else {
                    // Return the raw pointer - caller must call release_feature_set
                    Some(result)
                }
            } else {
                None
            }
        }
    }

    /// # Safety
    /// The caller must ensure that `features` is a valid pointer returned from a previous
    /// call to `process()` or `get_remaining_features()`.
    pub unsafe fn release_feature_set(&self, features: *mut VampFeatureList) {
        if let Some(release) = (*self.descriptor).releaseFeatureSet {
            release(features);
        }
    }

    pub fn get_remaining_features(&mut self) -> Option<*mut VampFeatureList> {
        unsafe {
            if let Some(get_remaining) = (*self.descriptor).getRemainingFeatures {
                let result = get_remaining(self.handle);
                if result.is_null() {
                    None
                } else {
                    // Return the raw pointer - caller must call release_feature_set
                    Some(result)
                }
            } else {
                None
            }
        }
    }

    pub fn reset(&mut self) {
        unsafe {
            if let Some(reset) = (*self.descriptor).reset {
                reset(self.handle);
            }
        }
    }

    pub fn set_parameter(&mut self, parameter: u32, value: f32) -> bool {
        unsafe {
            if let Some(set_parameter) = (*self.descriptor).setParameter {
                set_parameter(self.handle, parameter as i32, value);
                true
            } else {
                false
            }
        }
    }

    pub fn get_parameter(&self, parameter: u32) -> Option<f32> {
        unsafe {
            if let Some(get_parameter) = (*self.descriptor).getParameter {
                Some(get_parameter(self.handle, parameter as i32))
            } else {
                None
            }
        }
    }

    pub fn get_parameter_count(&self) -> u32 {
        unsafe { (*self.descriptor).parameterCount }
    }
}

impl Drop for VampPlugin {
    fn drop(&mut self) {
        unsafe {
            if let Some(cleanup) = (*self.descriptor).cleanup {
                cleanup(self.handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_creation() {
        let host = VampHost::new();
        let libraries = host.find_plugin_libraries();
        println!("Found {} plugin libraries", libraries.len());
    }

    #[test]
    fn test_library_loading() {
        let host = VampHost::new();
        let libraries = host.find_plugin_libraries();

        if let Some(first_lib) = libraries.first() {
            if let Some(library) = host.load_library(first_lib) {
                let plugins = library.list_plugins();
                println!(
                    "Found {} plugins in {}:",
                    plugins.len(),
                    first_lib.display()
                );
                for plugin in plugins.iter().take(3) {
                    println!("  {}: {}", plugin.identifier, plugin.name);
                }
            }
        }
    }

    #[test]
    fn test_plugin_instantiation() {
        let host = VampHost::new();
        let libraries = host.find_plugin_libraries();

        if let Some(first_lib) = libraries.first() {
            if let Some(library) = host.load_library(first_lib) {
                let plugins = library.list_plugins();
                if let Some(first_plugin) = plugins.first() {
                    if let Some(_plugin) = library.instantiate_plugin(first_plugin.index, 44100.0) {
                        println!(
                            "Successfully instantiated plugin: {}",
                            first_plugin.identifier
                        );
                    }
                }
            }
        }
    }
}
