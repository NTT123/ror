extern crate ror_sys;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use ror_sys::ffi::ONNXTensorElementDataType;
use ror_sys::ffi::OrtAllocatorType;
use ror_sys::ffi::OrtApi;
use ror_sys::ffi::OrtEnv;
pub use ror_sys::ffi::OrtLoggingLevel;
use ror_sys::ffi::OrtMemType;
use ror_sys::ffi::OrtSessionOptions;
use ror_sys::ffi::OrtStatus;
use ror_sys::ffi::OrtTensorTypeAndShapeInfo;
use ror_sys::ffi::ORT_API_VERSION;
use ror_sys::ffi::{OrtMemoryInfo, OrtSession, OrtValue};
use std::ffi::{c_void, CString};

#[derive(Debug, Clone, Copy)]
pub struct ROR {
    api: *const OrtApi,
    session: *const OrtSession,
}

unsafe impl Send for ROR {}
unsafe impl Sync for ROR {}

pub enum TensorData {
    FloatData(Vec<f32>),
    LongData(Vec<i32>),
}

pub use TensorData::FloatData;
pub use TensorData::LongData;

impl TensorData {
    pub fn into_f32_slice(&self) -> Option<&[f32]> {
        match self {
            FloatData(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn into_i32_slice(&self) -> Option<&[i32]> {
        match self {
            LongData(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

pub struct NamedTensor {
    name: String,
    data: TensorData,
    shape: Vec<i64>,
    mem_size: usize,
    dtype: ONNXTensorElementDataType,
}

impl NamedTensor {
    fn assert_correct_shape<T>(data: &[T], shape: &[i64]) {
        let num_elem = shape.iter().product::<i64>() as usize;
        if num_elem != data.len() {
            panic!(
                "Buffer size ({}) does not match with shape ({:?})!",
                num_elem, shape,
            );
        }
    }

    pub fn from_f32_slice(name: &str, data: &[f32], shape: &[i64]) -> Self {
        Self::assert_correct_shape(data, shape);
        let num_elem = shape.iter().product::<i64>() as usize;
        let mem_size: usize = num_elem * std::mem::size_of::<f32>();
        NamedTensor {
            name: String::from(name),
            data: FloatData(Vec::from(data)),
            shape: Vec::from(shape),
            mem_size,
            dtype: ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        }
    }

    pub fn from_i32_slice(name: &str, data: &[i32], shape: &[i64]) -> Self {
        Self::assert_correct_shape(data, shape);
        let num_elem = shape.iter().product::<i64>() as usize;
        let mem_size: usize = num_elem * std::mem::size_of::<i32>();
        NamedTensor {
            name: String::from(name),
            data: LongData(Vec::from(data)),
            shape: Vec::from(shape),
            mem_size,
            dtype: ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
        }
    }
}

pub struct InferenceOutput {
    pub data: TensorData,
    pub shape: Vec<i64>,
}

impl ROR {
    pub fn new(model_path: &str, logging_level: OrtLoggingLevel) -> Result<Self> {
        let api = Self::get_api()?;
        let env = Self::create_env(api, "rust-onnx-runtime", logging_level)?;
        let session_opts: *mut OrtSessionOptions = std::ptr::null_mut();
        let mut session: *mut OrtSession = std::ptr::null_mut();
        let model_path = CString::new(model_path).unwrap();
        let session = unsafe {
            let _ = (*api).CreateSession.context("CreateSession err")?(
                env,
                model_path.as_ptr(),
                session_opts,
                &mut session,
            );
            session
        };
        if session.is_null() {
            bail!("Cannot create ONNX session");
        };

        Ok(ROR { api, session })
    }

    pub fn run(
        self,
        inputs: &[NamedTensor],
        output_names: &[&str],
    ) -> Result<Vec<InferenceOutput>> {
        let input_names: Vec<CString> = inputs
            .iter()
            .map(|x| CString::new(x.name.as_str()).unwrap())
            .collect();
        let input_names_ptr: Vec<*const i8> = input_names.iter().map(|x| x.as_ptr()).collect();
        let c_output_names: Vec<CString> = output_names
            .iter()
            .map(|&x| CString::new(x).unwrap())
            .collect();
        let c_output_names_ptr: Vec<*const i8> =
            c_output_names.iter().map(|x| x.as_ptr()).collect();
        let input_values: Vec<*const OrtValue> =
            inputs.iter().map(|x| self.tensor_to_ort_value(x)).collect();
        let output_values = unsafe {
            let mut output_values: Vec<*mut OrtValue> =
                output_names.iter().map(|&_| std::ptr::null_mut()).collect();
            let status = (*self.api).Run.unwrap()(
                self.session as *mut OrtSession,
                std::ptr::null(),
                input_names_ptr.as_ptr(),
                input_values.as_ptr(),
                inputs.len(),
                c_output_names_ptr.as_ptr(),
                output_names.len(),
                output_values.as_mut_ptr(),
            );
            if !status.is_null() {
                bail!("{}", self.get_error_message(status).unwrap());
            }
            output_values
        };

        let outputs: Vec<InferenceOutput> = output_values
            .iter()
            .map(|&e| {
                let (shape, dtype) = self.get_value_shape_and_dtype(e);
                let size: i64 = shape.iter().product();
                let data = match dtype {
                    ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => {
                        let data = self.get_tensor_mutable_data(e) as *const i32;
                        let data =
                            unsafe { std::slice::from_raw_parts(data, size as usize) }.to_vec();
                        LongData(data)
                    }
                    ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
                        let data = self.get_tensor_mutable_data(e) as *const f32;
                        let data =
                            unsafe { std::slice::from_raw_parts(data, size as usize) }.to_vec();
                        FloatData(data)
                    }
                    _ => {
                        panic!("Not supported data type!");
                    }
                };

                InferenceOutput { data, shape }
            })
            .collect();

        unsafe {
            input_values.iter().for_each(|&x| {
                (*self.api).ReleaseValue.unwrap()(x as *mut OrtValue);
            });
            output_values.iter().for_each(|&x| {
                (*self.api).ReleaseValue.unwrap()(x as *mut OrtValue);
            });
        }

        Ok(outputs)
    }

    fn get_api() -> Result<*const OrtApi> {
        let ort_api_base = unsafe {
            ror_sys::ffi::OrtGetApiBase()
                .as_ref()
                .context("Cannot get api base")?
        };
        let api = unsafe { ort_api_base.GetApi.context("Cannot call GetAPI")?(ORT_API_VERSION) };
        Ok(api)
    }

    fn create_env(
        api: *const OrtApi,
        name: &str,
        logging_level: OrtLoggingLevel,
    ) -> Result<*mut OrtEnv> {
        let cname = CString::new(name)?;
        let env = unsafe {
            let create_env_fn = (*api).CreateEnv.context("Cannot call CreateEnv")?;
            let mut env: *mut OrtEnv = std::ptr::null_mut();
            create_env_fn(logging_level, cname.as_ptr(), &mut env);
            env
        };
        Ok(env)
    }

    fn tensor_to_ort_value(self, tensor: &NamedTensor) -> *const OrtValue {
        let mem_info = self.create_cpu_memory_info();
        let data_ptr = match &tensor.data {
            FloatData(d) => d.as_ptr() as *mut c_void,
            LongData(d) => d.as_ptr() as *mut c_void,
        };
        let mut ort_value: *mut OrtValue = std::ptr::null_mut();
        unsafe {
            (*self.api).CreateTensorWithDataAsOrtValue.unwrap()(
                mem_info,
                data_ptr,
                tensor.mem_size,
                tensor.shape.as_ptr(),
                tensor.shape.len(),
                tensor.dtype,
                &mut ort_value,
            );
            if ort_value.is_null() {
                panic!("Cannot create ort value");
            }
            (*self.api).ReleaseMemoryInfo.unwrap()(mem_info);
            ort_value
        }
    }

    fn create_cpu_memory_info(self) -> *mut OrtMemoryInfo {
        unsafe {
            let mut mem_info: *mut OrtMemoryInfo = std::ptr::null_mut();

            (*self.api).CreateCpuMemoryInfo.unwrap()(
                OrtAllocatorType::OrtArenaAllocator,
                OrtMemType::OrtMemTypeDefault,
                &mut mem_info,
            );
            if mem_info.is_null() {
                panic!("Cannot create cpu mem info");
            }
            mem_info
        }
    }

    fn get_tensor_mutable_data(self, x: *const OrtValue) -> *const c_void {
        unsafe {
            let mut output: *mut c_void = std::ptr::null_mut();
            (*self.api).GetTensorMutableData.unwrap()(x as *mut OrtValue, &mut output);
            output
        }
    }

    fn get_value_shape_and_dtype(
        self,
        value: *const OrtValue,
    ) -> (Vec<i64>, ONNXTensorElementDataType) {
        let mut type_and_shape_info: *mut OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let type_and_shape_info = unsafe {
            (*self.api).GetTensorTypeAndShape.unwrap()(value, &mut type_and_shape_info);
            type_and_shape_info
        };

        let dtype = unsafe {
            let mut dtype: ONNXTensorElementDataType =
                ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            (*self.api).GetTensorElementType.unwrap()(type_and_shape_info, &mut dtype);
            dtype
        };

        let num_dim = unsafe {
            let mut _num_dim = 0;
            (*self.api).GetDimensionsCount.expect("Err")(type_and_shape_info, &mut _num_dim);
            _num_dim
        };
        let dims = unsafe {
            let mut _dims: Vec<i64> = vec![0; num_dim];
            (*self.api).GetDimensions.unwrap()(type_and_shape_info, _dims.as_mut_ptr(), num_dim);
            _dims
        };
        unsafe {
            (*self.api).ReleaseTensorTypeAndShapeInfo.unwrap()(type_and_shape_info);
        };
        (dims, dtype)
    }

    fn get_error_message(self, status: *const OrtStatus) -> Result<String> {
        if status.is_null() {
            bail!("No error!")
        } else {
            let msg = unsafe { (*self.api).GetErrorMessage.unwrap()(status) };
            let msg = unsafe { std::ffi::CStr::from_ptr(msg).to_str() };
            msg.map(|e| e.to_string()).context("Utf8 error")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let ror = ROR::new(
            "models/mnist.onnx",
            OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
        )
        .unwrap();
        for _ in 1..10 {
            let o = ror
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
        let ror = ROR::new(
            "models/mnist.onnx",
            OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
        )
        .unwrap();
        let o = ror
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
        let l1_dist = predict
            .iter()
            .zip(target.iter())
            .map(|(&x, &y)| (x - y).abs())
            .reduce(f32::max)
            .unwrap();
        assert!(l1_dist < 1e-8);
    }

    #[test]
    fn test_mnist_nonzero_input() {
        let ror = ROR::new(
            "models/mnist.onnx",
            OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
        )
        .unwrap();
        let o = ror
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
        let l1_dist = predict
            .iter()
            .zip(target.iter())
            .map(|(&x, &y)| (x - y).abs())
            .reduce(f32::max)
            .unwrap();
        assert!(l1_dist < 1e-6);
    }

    #[test]
    fn test_mnist_nonzero_input_multithread() {
        let ror = ROR::new(
            "models/mnist.onnx",
            OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
        )
        .unwrap();

        let mut handles = Vec::with_capacity(1000);
        for _ in 0..100 {
            let handle = std::thread::spawn(move || {
                for _ in 0..10000 {
                    let o = ror
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
                    let l1_dist = predict
                        .iter()
                        .zip(target.iter())
                        .map(|(&x, &y)| (x - y).abs())
                        .reduce(f32::max)
                        .unwrap();
                    assert!(l1_dist < 1e-6);
                }
            });
            handles.push(handle);
        }
        for handle in handles {
            handle.join().unwrap();
        }
    }
}
