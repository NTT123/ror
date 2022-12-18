use std::env;
use std::path::PathBuf;

fn main() {
    let t = std::env::var("ONNX_DIR").unwrap_or(String::from(
        "/opt/onnxruntime-linux-x64-1.13.1",
    ));
    let onnx_dir = PathBuf::from(t);
    println!("cargo:rustc-link-search={}", onnx_dir.join("lib").display());
    println!("cargo:rustc-link-lib=onnxruntime");
    let bindings = bindgen::Builder::default()
        .header(
            onnx_dir
                .join("include/onnxruntime_c_api.h")
                .to_str()
                .unwrap(),
        )
        .default_enum_style(bindgen::EnumVariation::Rust{non_exhaustive: false})
        .allowlist_var("ORT_API_VERSION")
        .allowlist_function("OrtGetApiBase")
        .allowlist_function("GetTensorTypeAndShape")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
