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

pub struct Tensor<T> {
    name: String,
    data: Vec<T>,
    shape: Vec<i64>,
}

impl<T: Clone> Tensor<T> {
    pub fn new(name: &str, data: &[T], shape: &[i64]) -> Self {
        Tensor::<T> {
            name: String::from(name),
            data: Vec::from(data),
            shape: Vec::from(shape),
        }
    }
}

pub struct InferenceOutput<T> {
    pub data: Vec<T>,
    pub shape: Vec<i64>,
}

impl ROR {
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
    ) -> Result<&'static mut OrtEnv> {
        let cname = CString::new(name)?;
        let env = unsafe {
            let create_env_fn = (*api).CreateEnv.context("Cannot call CreateEnv")?;
            let mut env: *mut OrtEnv = std::ptr::null_mut();
            create_env_fn(logging_level, cname.as_ptr(), &mut env);
            env.as_mut().context("Cannot create env")?
        };
        Ok(env)
    }

    fn tensor_to_ort_value<T>(self, tensor: &Tensor<T>) -> *const OrtValue {
        let mem_info = self.create_cpu_memory_info();
        let mut ort_value: *mut OrtValue = std::ptr::null_mut();
        unsafe {
            (*self.api).CreateTensorWithDataAsOrtValue.unwrap()(
                mem_info,
                tensor.data.as_ptr() as *mut c_void,
                tensor.data.len() * std::mem::size_of::<T>(),
                tensor.shape.as_ptr(),
                tensor.shape.len(),
                ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &mut ort_value,
            );
            if ort_value.is_null() {
                panic!("Cannot create ort value");
            }
            ort_value
        }
    }

    fn create_cpu_memory_info(self) -> *const OrtMemoryInfo {
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

    pub fn get_value_shape(self, value: *const OrtValue) -> Vec<i64> {
        let mut type_and_shape_info: *mut OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let type_and_shape_info = unsafe {
            (*self.api).GetTensorTypeAndShape.unwrap()(value, &mut type_and_shape_info);
            type_and_shape_info
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
        dims
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

    pub fn run(
        self,
        inputs: Vec<Tensor<f32>>,
        output_names: &[String],
    ) -> Result<Vec<InferenceOutput<f32>>> {
        let input_names: Vec<CString> = inputs
            .iter()
            .map(|x| CString::new(x.name.as_str()).unwrap())
            .collect();
        let input_names_ptr: Vec<*const i8> = input_names.iter().map(|x| x.as_ptr()).collect();
        let c_output_names: Vec<CString> = output_names
            .iter()
            .map(|x| CString::new(x.as_str()).unwrap())
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

        let outputs: Vec<InferenceOutput<f32>> = output_values
            .iter()
            .map(|&e| {
                let data = self.get_tensor_mutable_data(e) as *const f32;
                let shape = self.get_value_shape(e);
                let size: i64 = shape.iter().product();
                let data = unsafe { std::slice::from_raw_parts(data, size as usize) };
                InferenceOutput {
                    data: data.to_vec(),
                    shape,
                }
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
        let inp: [f32; 28 * 28] = [0f32; 28 * 28];
        let shape: Vec<i64> = vec![1, 1, 28, 28];
        let output_names: Vec<String> = vec![String::from("Plus214_Output_0")];
        for _ in 1..10 {
            let o = ror
                .run(
                    vec![Tensor::<f32>::new("Input3", &inp.to_vec(), &shape)],
                    &output_names,
                )
                .unwrap();
            assert_eq!(o[0].shape, [1, 10]);
        }
    }
}
