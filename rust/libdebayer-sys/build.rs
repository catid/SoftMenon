use anyhow::{Context, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    let mut clang_args = Vec::new();
    if std::env::var_os("DOCS_RS").is_some() {
        // Generate docs from the real API and a declaration-only CUDA header.
        clang_args.push("-Idoc-headers".to_owned());
    } else {
        let library = pkg_config::probe_library("libdebayer")?;
        clang_args.extend(
            library
                .include_paths
                .iter()
                .map(|path| format!("-I{}", path.display())),
        );
        let cuda = PathBuf::from(
            std::env::var_os("CUDA_PATH").context("CUDA_PATH must name the CUDA toolkit root")?,
        );
        clang_args.push(format!("-I{}", cuda.join("include").display()));
        // pkg-config already emits libdebayer's link directives. CUDA_PATH is
        // a toolkit root (including in flake.nix), not its library directory.
        for directory in [
            "lib64",
            "lib",
            "targets/x86_64-linux/lib",
            "targets/aarch64-linux/lib",
        ] {
            let directory = cuda.join(directory);
            if directory.is_dir() {
                println!("cargo:rustc-link-search=native={}", directory.display());
            }
        }
        println!("cargo:rustc-link-lib=cudart");
    }
    // The sys crate packages the public header so docs.rs needs no installed C library.
    let bindings = bindgen::Builder::default()
        .clang_args(clang_args)
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .default_enum_style(bindgen::EnumVariation::ModuleConsts)
        .size_t_is_usize(true)
        .allowlist_function("debayer_.*")
        .allowlist_function("cuda(MallocPitch|Memset2D|Memcpy2DAsync|StreamCreate|StreamDestroy|StreamSynchronize|Free|GetLastError)")
        .allowlist_var("SARONIC_DEBAYER_PAD|KERNEL_BLOCK_SIZE")
        .generate().context("Unable to generate libdebayer/CUDA bindings")?;
    let output = PathBuf::from(std::env::var_os("OUT_DIR").context("missing OUT_DIR")?);
    bindings.write_to_file(output.join("bindings.rs"))?;
    Ok(())
}
