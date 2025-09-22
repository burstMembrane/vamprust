use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let submodule_path = Path::new(&manifest_dir).join("vamp-plugin-sdk");

    // Check if we should use the submodule or system-installed SDK
    let use_submodule = submodule_path.exists() && submodule_path.join("vamp").exists();

    if use_submodule {
        println!("cargo:warning=Using bundled Vamp SDK submodule");
        build_submodule(&submodule_path);
        configure_submodule_build(&submodule_path);
    } else {
        println!("cargo:warning=Using system-installed Vamp SDK");
        configure_system_build();
    }

    generate_bindings(&submodule_path, use_submodule);
}

fn build_submodule(submodule_path: &Path) {
    let build_dir = submodule_path.join("build");

    // Create build directory
    std::fs::create_dir_all(&build_dir).expect("Failed to create build directory");

    // Run cmake configure
    let mut cmake_cmd = Command::new("cmake");
    cmake_cmd
        .current_dir(&build_dir)
        .arg("..")
        .arg("-DVAMPSDK_BUILD_SIMPLE_HOST=ON")
        .arg("-DCMAKE_BUILD_TYPE=Release");

    // Add platform-specific flags
    #[cfg(target_os = "macos")]
    {
        cmake_cmd.arg("-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15");
    }

    let output = cmake_cmd.output().expect("Failed to run cmake configure");
    if !output.status.success() {
        panic!(
            "CMake configure failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Run cmake build
    let output = Command::new("cmake")
        .current_dir(&build_dir)
        .arg("--build")
        .arg(".")
        .arg("--config")
        .arg("Release")
        .output()
        .expect("Failed to run cmake build");

    if !output.status.success() {
        panic!(
            "CMake build failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn configure_submodule_build(submodule_path: &Path) {
    let build_dir = submodule_path.join("build");

    // Check for different possible lib locations
    let possible_lib_dirs = [
        build_dir.join("lib"),
        build_dir.join("src").join("vamp-hostsdk"),
        build_dir.join("src").join("vamp-sdk"),
        build_dir.clone(),
    ];

    for lib_dir in &possible_lib_dirs {
        if lib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }
    }

    // Link libraries
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=static=vamp-hostsdk");
        println!("cargo:rustc-link-lib=static=vamp-sdk");
        println!("cargo:rustc-link-lib=c++");
    }

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=static=vamp-hostsdk");
        println!("cargo:rustc-link-lib=static=vamp-sdk");
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=dl");
    }

    // Rerun if submodule changes
    println!("cargo:rerun-if-changed={}", submodule_path.display());
}

fn configure_system_build() {
    // Try pkg-config first for system-installed SDK
    let use_pkg_config = pkg_config::Config::new().probe("vamp-hostsdk").is_ok();

    if !use_pkg_config {
        // Manual fallback - assume SDK is installed in standard locations
        // or user has set VAMP_SDK_PATH environment variable
        if let Ok(sdk_path) = env::var("VAMP_SDK_PATH") {
            println!("cargo:rustc-link-search=native={}/lib", sdk_path);
        } else {
            // Try common installation paths
            let common_paths = [
                "/usr/local",
                "/usr",
                "/opt/homebrew", // macOS Homebrew ARM
            ];

            for path in &common_paths {
                let lib_path = format!("{}/lib", path);
                if Path::new(&lib_path).exists() {
                    println!("cargo:rustc-link-search=native={}", lib_path);
                }
            }
        }

        // Link the required libraries
        #[cfg(target_os = "macos")]
        {
            println!("cargo:rustc-link-lib=vamp-hostsdk");
            println!("cargo:rustc-link-lib=vamp-sdk");
            println!("cargo:rustc-link-lib=c++");
        }

        #[cfg(target_os = "linux")]
        {
            println!("cargo:rustc-link-lib=vamp-hostsdk");
            println!("cargo:rustc-link-lib=vamp-sdk");
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=dl");
        }
    }
}

fn generate_bindings(submodule_path: &Path, use_submodule: bool) {
    // Generate bindings for the core Vamp C API only
    let mut builder = bindgen::Builder::default()
        // Include only the core vamp.h header
        .header_contents(
            "wrapper.h",
            r#"
            #include "vamp/vamp.h"
        "#,
        )
        // Allowlist the core Vamp functions and types
        .allowlist_function("vampGetPluginDescriptor")
        .allowlist_type("Vamp.*")
        .allowlist_type("_Vamp.*")
        .allowlist_var("VAMP_API_VERSION")
        // Generate helpful traits
        .derive_debug(true)
        .derive_default(true)
        .derive_copy(true)
        // Handle C naming conventions
        .prepend_enum_name(false)
        .generate_comments(false);

    // Add include paths
    if use_submodule {
        builder = builder.clang_arg(format!("-I{}", submodule_path.display()));
    } else if let Ok(sdk_path) = env::var("VAMP_SDK_PATH") {
        builder = builder.clang_arg(format!("-I{}/include", sdk_path));
    } else {
        // Try standard include paths
        let include_paths = [
            "/usr/local/include",
            "/usr/include",
            "/opt/homebrew/include",
        ];

        for path in &include_paths {
            if Path::new(path).exists() {
                builder = builder.clang_arg(format!("-I{}", path));
            }
        }
    }

    // Generate the bindings
    let bindings = builder
        .generate()
        .expect("Unable to generate bindings for Vamp SDK");

    // Write the bindings to $OUT_DIR/bindings.rs
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VAMP_SDK_PATH");
}
