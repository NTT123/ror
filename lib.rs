extern crate ror_sys;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
pub use ror_sys::ffi::GraphOptimizationLevel;
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

#[derive(Debug, Clone, Copy)]
pub struct Api {
    raw_ptr: *const OrtApi,
}

impl Api {
    pub fn new() -> Result<Self> {
        let ort_api_base = unsafe {
            ror_sys::ffi::OrtGetApiBase()
                .as_ref()
                .context("Cannot get api base")?
        };
        let api = unsafe { ort_api_base.GetApi.context("Cannot call GetAPI")?(ORT_API_VERSION) };
        Ok(Self { raw_ptr: api })
    }

    pub fn create_env(&self, name: &str, logging_level: OrtLoggingLevel) -> Result<Env> {
        Env::new(&self, name, logging_level)
    }
}

#[derive(Debug, Clone)]
pub struct Session {
    raw_api_ptr: *const OrtApi,
    raw_session_ptr: *mut OrtSession,
    _env: Env,
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { (*self.raw_api_ptr).ReleaseSession.unwrap()(self.raw_session_ptr) }
    }
}

#[derive(Default)]
pub struct SessionOptions {
    inter_op_num_threads: Option<i32>,
    intra_op_num_threads: Option<i32>,
    graph_optimization_level: Option<ror_sys::ffi::GraphOptimizationLevel>,
}

impl SessionOptions {
    pub fn new() -> Self {
        SessionOptions {
            inter_op_num_threads: None,
            intra_op_num_threads: None,
            graph_optimization_level: None,
        }
    }

    pub fn set_inter_op_num_threads(&self, num_threads: i32) -> Self {
        SessionOptions {
            inter_op_num_threads: Some(num_threads),
            intra_op_num_threads: self.intra_op_num_threads,
            graph_optimization_level: self.graph_optimization_level,
        }
    }

    pub fn set_intra_op_num_threads(&self, num_threads: i32) -> Self {
        SessionOptions {
            intra_op_num_threads: Some(num_threads),
            inter_op_num_threads: self.intra_op_num_threads,
            graph_optimization_level: self.graph_optimization_level,
        }
    }

    pub fn set_graph_optimization_level(&self, level: GraphOptimizationLevel) -> Self {
        SessionOptions {
            intra_op_num_threads: self.intra_op_num_threads,
            inter_op_num_threads: self.intra_op_num_threads,
            graph_optimization_level: Some(level),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Env {
    raw_api_ptr: *const OrtApi,
    raw_env_ptr: *mut OrtEnv,
}

impl Env {
    pub fn new(api: &Api, name: &str, logging_level: OrtLoggingLevel) -> Result<Self> {
        let cname = CString::new(name)?;
        let env = unsafe {
            let create_env_fn = (*api.raw_ptr).CreateEnv.context("Cannot call CreateEnv")?;
            let mut env: *mut OrtEnv = std::ptr::null_mut();
            create_env_fn(logging_level, cname.as_ptr(), &mut env);
            env
        };
        Ok(Env {
            raw_api_ptr: api.raw_ptr,
            raw_env_ptr: env,
        })
    }

    pub fn create_session_with_options(
        self,
        model_path: &str,
        options: SessionOptions,
    ) -> Result<Session> {
        let model_path = CString::new(model_path).unwrap();
        let session_opts = unsafe {
            let mut session_opts: *mut OrtSessionOptions = std::ptr::null_mut();
            (*self.raw_api_ptr).CreateSessionOptions.unwrap()(&mut session_opts);
            if let Some(n) = options.inter_op_num_threads {
                (*self.raw_api_ptr).SetInterOpNumThreads.unwrap()(session_opts, n);
            }
            if let Some(n) = options.intra_op_num_threads {
                (*self.raw_api_ptr).SetIntraOpNumThreads.unwrap()(session_opts, n);
            }
            if let Some(l) = options.graph_optimization_level {
                (*self.raw_api_ptr)
                    .SetSessionGraphOptimizationLevel
                    .unwrap()(session_opts, l);
            }
            session_opts
        };
        let session_ptr = unsafe {
            let mut session_ptr: *mut OrtSession = std::ptr::null_mut();
            let _ = (*self.raw_api_ptr)
                .CreateSession
                .context("CreateSession err")?(
                self.raw_env_ptr,
                model_path.as_ptr(),
                session_opts,
                &mut session_ptr,
            );
            session_ptr
        };
        unsafe { (*self.raw_api_ptr).ReleaseSessionOptions.unwrap()(session_opts) };
        if session_ptr.is_null() {
            bail!("Cannot create ONNX session");
        };

        Ok(Session {
            raw_api_ptr: self.raw_api_ptr,
            raw_session_ptr: session_ptr,
            _env: self,
        })
    }

    pub fn create_session(self, model_path: &str) -> Result<Session> {
        self.create_session_with_options(model_path, SessionOptions::new())
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        unsafe { (*self.raw_api_ptr).ReleaseEnv.unwrap()(self.raw_env_ptr) }
    }
}

pub struct InferenceOutput {
    pub data: TensorData,
    pub shape: Vec<i64>,
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

struct MemoryInfo {
    raw_api_ptr: *const OrtApi,
    raw_ptr: *mut OrtMemoryInfo,
}

impl MemoryInfo {
    fn new(raw_api_ptr: *const OrtApi) -> Self {
        unsafe {
            let mut mem_info: *mut OrtMemoryInfo = std::ptr::null_mut();

            (*raw_api_ptr).CreateCpuMemoryInfo.unwrap()(
                OrtAllocatorType::OrtArenaAllocator,
                OrtMemType::OrtMemTypeDefault,
                &mut mem_info,
            );
            if mem_info.is_null() {
                panic!("Cannot create cpu mem info");
            }
            Self {
                raw_api_ptr,
                raw_ptr: mem_info,
            }
        }
    }
}

impl Drop for MemoryInfo {
    fn drop(&mut self) {
        unsafe { (*self.raw_api_ptr).ReleaseMemoryInfo.unwrap()(self.raw_ptr) };
    }
}
struct Value {
    raw_ptr: *mut OrtValue,
    raw_api_ptr: *const OrtApi,
}

impl Value {
    fn from_tensor(raw_api_ptr: *const OrtApi, tensor: &NamedTensor) -> Self {
        let mem_info = MemoryInfo::new(raw_api_ptr);
        let data_ptr = match &tensor.data {
            FloatData(d) => d.as_ptr() as *mut c_void,
            LongData(d) => d.as_ptr() as *mut c_void,
        };
        let mut ort_value: *mut OrtValue = std::ptr::null_mut();
        unsafe {
            (*raw_api_ptr).CreateTensorWithDataAsOrtValue.unwrap()(
                mem_info.raw_ptr,
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
            Value {
                raw_ptr: ort_value,
                raw_api_ptr: raw_api_ptr,
            }
        }
    }

    fn get_value_shape_and_dtype(&self) -> (Vec<i64>, ONNXTensorElementDataType) {
        let mut type_and_shape_info: *mut OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let type_and_shape_info = unsafe {
            (*self.raw_api_ptr).GetTensorTypeAndShape.unwrap()(
                self.raw_ptr,
                &mut type_and_shape_info,
            );
            type_and_shape_info
        };

        let dtype = unsafe {
            let mut dtype: ONNXTensorElementDataType =
                ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            (*self.raw_api_ptr).GetTensorElementType.unwrap()(type_and_shape_info, &mut dtype);
            dtype
        };

        let num_dim = unsafe {
            let mut _num_dim = 0;
            (*self.raw_api_ptr).GetDimensionsCount.expect("Err")(
                type_and_shape_info,
                &mut _num_dim,
            );
            _num_dim
        };
        let dims = unsafe {
            let mut _dims: Vec<i64> = vec![0; num_dim];
            (*self.raw_api_ptr).GetDimensions.unwrap()(
                type_and_shape_info,
                _dims.as_mut_ptr(),
                num_dim,
            );
            _dims
        };
        unsafe {
            (*self.raw_api_ptr).ReleaseTensorTypeAndShapeInfo.unwrap()(type_and_shape_info);
        };
        (dims, dtype)
    }

    fn get_tensor_mutable_data(&self) -> *const c_void {
        unsafe {
            let mut output: *mut c_void = std::ptr::null_mut();
            (*self.raw_api_ptr).GetTensorMutableData.unwrap()(self.raw_ptr, &mut output);
            output
        }
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        unsafe { (*self.raw_api_ptr).ReleaseValue.unwrap()(self.raw_ptr) };
    }
}

impl Session {
    pub fn run(
        &self,
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
        let input_values: Vec<Value> = inputs
            .iter()
            .map(|x| Value::from_tensor(self.raw_api_ptr, x))
            .collect();
        let input_values_ptr: Vec<*const OrtValue> = input_values
            .iter()
            .map(|x| x.raw_ptr as *const OrtValue)
            .collect();
        let output_values: Vec<Value> = unsafe {
            let mut output_values: Vec<*mut OrtValue> =
                output_names.iter().map(|&_| std::ptr::null_mut()).collect();
            let status = (*self.raw_api_ptr).Run.unwrap()(
                self.raw_session_ptr as *mut OrtSession,
                std::ptr::null(),
                input_names_ptr.as_ptr(),
                input_values_ptr.as_ptr(),
                inputs.len(),
                c_output_names_ptr.as_ptr(),
                output_names.len(),
                output_values.as_mut_ptr(),
            );
            if !status.is_null() {
                bail!("{}", self.get_error_message(status).unwrap());
            }
            output_values
                .iter()
                .map(|&x| Value {
                    raw_ptr: x,
                    raw_api_ptr: self.raw_api_ptr,
                })
                .collect()
        };

        let outputs: Vec<InferenceOutput> = output_values
            .iter()
            .map(|e| {
                let (shape, dtype) = e.get_value_shape_and_dtype();
                let size: i64 = shape.iter().product();
                let data = match dtype {
                    ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => {
                        let data = e.get_tensor_mutable_data() as *const i32;
                        let data =
                            unsafe { std::slice::from_raw_parts(data, size as usize) }.to_vec();
                        LongData(data)
                    }
                    ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
                        let data = e.get_tensor_mutable_data() as *const f32;
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
        Ok(outputs)
    }

    fn get_error_message(&self, status: *const OrtStatus) -> Result<String> {
        if status.is_null() {
            bail!("No error!")
        } else {
            let msg = unsafe { (*self.raw_api_ptr).GetErrorMessage.unwrap()(status) };
            let msg = unsafe { std::ffi::CStr::from_ptr(msg).to_str() };
            msg.map(|e| e.to_string()).context("Utf8 error")
        }
    }
}
