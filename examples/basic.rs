use anyhow::{Context, Result};
use ror::NamedTensor;

fn main() -> Result<()> {
    let session = ror::Api::new()?
        .create_env(
            "onnx-env-1",
            ror::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
        )?
        .create_session("models/mnist.onnx")?;
    let inp: [f32; 28 * 28] = [0f32; 28 * 28];
    let shape: Vec<i64> = vec![1, 1, 28, 28];
    let o = session.run(
        &[NamedTensor::from_f32_slice("Input3", &inp, &shape)],
        &["Plus214_Output_0"],
    )?;

    println!("output {:?}", o[0].data.into_f32_slice().context("Err")?);
    println!("shape  {:?}", o[0].shape);
    Ok(())
}
