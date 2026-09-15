use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=ARCQML_RUNTIME_LIB_DIR");
    if env::var_os("CARGO_FEATURE_PRIVATE_SOURCE").is_some() {
        return;
    }

    let target = env::var("TARGET").expect("Cargo must define TARGET");
    let library_dir = env::var_os("ARCQML_RUNTIME_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("native").join(&target));
    if !library_dir.is_dir() {
        panic!(
            "ArcQML Runtime for {target} was not found at {}. Install the matching Runtime SDK or set ARCQML_RUNTIME_LIB_DIR.",
            library_dir.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", library_dir.display());
    println!("cargo:rustc-link-lib=static=arcqml_runtime_private");
}
