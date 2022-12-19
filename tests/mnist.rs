#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ror::*;

    #[test]
    fn it_works() {
        let env = ror::Api::new()
            .unwrap()
            .create_env(
                "onnx-env-1",
                ror::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            )
            .unwrap();
        let session = env.create_session("models/mnist.onnx").unwrap();
        for _ in 1..10 {
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
            assert_eq!(o[0].shape, [1, 10]);
        }
    }

    #[test]
    fn test_mnist_zero_input() {
        let env = ror::Api::new()
            .unwrap()
            .create_env(
                "onnx-env-1",
                ror::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            )
            .unwrap();
        let session = env.create_session("models/mnist.onnx").unwrap();
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
    fn test_mnist_nonzero_input() {
        let env = ror::Api::new()
            .unwrap()
            .create_env(
                "onnx-env-1",
                ror::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            )
            .unwrap();
        let session = env.create_session("models/mnist.onnx").unwrap();
        let o = session
            .run(
                &[NamedTensor::from_f32_slice(
                    "Input3",
                    &[1.0f32; 28 * 28],
                    &[1, 1, 28, 28],
                )],
                &["Plus214_Output_0"],
            )
            .unwrap();

        let predict = o[0].data.into_f32_slice().unwrap();
        let target: [f32; 10] = [
            -1.7834078073501587,
            -1.5465246438980103,
            -0.6232262253761292,
            0.8309603333473206,
            -1.6586133241653442,
            1.6030707359313965,
            2.7646830081939697,
            -3.4099056720733643,
            1.6252055168151855,
            -0.16226638853549957,
        ];
        let dist = predict
            .iter()
            .zip(target.iter())
            .map(|(&x, &y)| (x - y).abs())
            .reduce(f32::max)
            .unwrap();
        assert!(dist < 1e-6);
    }

    #[test]
    fn test_mnist_nonzero_input_multithread() {
        let env = ror::Api::new()
            .unwrap()
            .create_env(
                "onnx-env-1",
                ror::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            )
            .unwrap();
        let session = env.create_session("models/mnist.onnx").unwrap();
        let session = Arc::new(session);
        let mut handles = Vec::with_capacity(1000);
        for _ in 0..100 {
            let session = Arc::clone(&session);
            let handle = std::thread::spawn(move || {
                for _ in 0..10000 {
                    let o = session
                        .run(
                            &[NamedTensor::from_f32_slice(
                                "Input3",
                                &[1.0f32; 28 * 28],
                                &[1, 1, 28, 28],
                            )],
                            &["Plus214_Output_0"],
                        )
                        .unwrap();

                    let predict = o[0].data.into_f32_slice().unwrap();
                    let target: [f32; 10] = [
                        -1.7834078073501587,
                        -1.5465246438980103,
                        -0.6232262253761292,
                        0.8309603333473206,
                        -1.6586133241653442,
                        1.6030707359313965,
                        2.7646830081939697,
                        -3.4099056720733643,
                        1.6252055168151855,
                        -0.16226638853549957,
                    ];
                    let dist = predict
                        .iter()
                        .zip(target.iter())
                        .map(|(&x, &y)| (x - y).abs())
                        .reduce(f32::max)
                        .unwrap();
                    assert!(dist < 1e-6);
                }
            });
            handles.push(handle);
        }
        for handle in handles {
            handle.join().unwrap();
        }
    }
}
