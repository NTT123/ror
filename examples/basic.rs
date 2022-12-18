use anyhow::Result;
use ror::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
use ror::Tensor;
use ror::ROR;

fn main() -> Result<()> {
    let runner = ROR::new("models/mnist.onnx", ORT_LOGGING_LEVEL_WARNING)?;
    let inp: [f32; 28 * 28] = [0f32; 28 * 28];
    let shape: Vec<i64> = vec![1, 1, 28, 28];
    let output_names: Vec<String> = vec![String::from("Plus214_Output_0")];
    let o = runner.run(
        vec![Tensor::<f32>::new("Input3", &inp.to_vec(), &shape)],
        &output_names,
    )?;

    println!("output {:?}", o[0].data);
    println!("shape  {:?}", o[0].shape);
    Ok(())
}
