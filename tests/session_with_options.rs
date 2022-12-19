#[cfg(test)]
mod tests {
    use ror::*;

    #[test]
    fn test_session_options_graph_optim_level() {
        let env = ror::Api::new()
            .unwrap()
            .create_env(
                "onnx-env-1",
                ror::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            )
            .unwrap();
        let options: SessionOptions = SessionOptions::new()
            .set_graph_optimization_level(ror::GraphOptimizationLevel::ORT_ENABLE_BASIC);
        let session = env
            .create_session_with_options("models/mnist.onnx", options)
            .unwrap();
        let o = session
            .run(
                &[NamedTensor::from_f32_slice(
                    "Input3",
                    &[0f32; 28 * 28],
                    &[1, 1, 28, 28],
                )],
                &["Plus214_Output_0"],
            )
            .unwrap();

        let predict = o[0].data.into_f32_slice().unwrap();
        let target: [f32; 10] = [
            -0.04485603,
            0.00779166,
            0.06810082,
            0.02999374,
            -0.12640963,
            0.14021875,
            -0.0552849,
            -0.04938382,
            0.08432205,
            -0.05454041,
        ];
        let dist = predict
            .iter()
            .zip(target.iter())
            .map(|(&x, &y)| (x - y).abs())
            .reduce(f32::max)
            .unwrap();
        assert!(dist < 1e-8);
    }

    #[test]
    fn test_session_options_num_threads() {
        let env = ror::Api::new()
            .unwrap()
            .create_env(
                "onnx-env-1",
                ror::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            )
            .unwrap();
        let options: SessionOptions = SessionOptions::new()
            .set_inter_op_num_threads(4)
            .set_intra_op_num_threads(2);
        let session = env
            .create_session_with_options("models/mnist.onnx", options)
            .unwrap();
        let o = session
            .run(
                &[NamedTensor::from_f32_slice(
                    "Input3",
                    &[0f32; 28 * 28],
                    &[1, 1, 28, 28],
                )],
                &["Plus214_Output_0"],
            )
            .unwrap();

        let predict = o[0].data.into_f32_slice().unwrap();
        let target: [f32; 10] = [
            -0.04485603,
            0.00779166,
            0.06810082,
            0.02999374,
            -0.12640963,
            0.14021875,
            -0.0552849,
            -0.04938382,
            0.08432205,
            -0.05454041,
        ];
        let dist = predict
            .iter()
            .zip(target.iter())
            .map(|(&x, &y)| (x - y).abs())
            .reduce(f32::max)
            .unwrap();
        assert!(dist < 1e-8);
    }
}
