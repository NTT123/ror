extern crate ror_sys;
use ror_sys::ffi::OrtApi;
use ror_sys::ffi::OrtEnv;
use ror_sys::ffi::OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE;
use ror_sys::ffi::OrtSession;
use ror_sys::ffi::OrtSessionOptions;
use ror_sys::ffi::ORT_API_VERSION;
use std::ffi::CString;

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct ROR {
    api: *const OrtApi,
    env: *const OrtEnv,
    session: *const OrtSession,
}

impl ROR {
    pub fn new(model_path: &str) -> Self {
        let api = unsafe {
            let ort_api_base = *ror_sys::ffi::OrtGetApiBase();
            let get_api_fn = ort_api_base.GetApi.expect("GetApi err");
            let api = get_api_fn(ORT_API_VERSION);
            api
        };
        if api.is_null() {
            panic!("Cannot create ONNX runtime API");
        };

        let mut env: *mut OrtEnv = std::ptr::null_mut();
        let env = unsafe {
            let name = CString::new("rust-onnx-runtime").unwrap();
            (*api).CreateEnv.expect("CreateEnv err")(
                OrtLoggingLevel_ORT_LOGGING_LEVEL_VERBOSE,
                name.as_ptr(),
                &mut env,
            );
            env
        };
        if env.is_null() {
            panic!("Cannot create ONNX env");
        };

        let mut session_opts: *mut OrtSessionOptions = std::ptr::null_mut();
        let session_opts = unsafe {
            (*api)
                .CreateSessionOptions
                .expect("CreateSessionOptions err")(&mut session_opts);
            session_opts
        };
        if session_opts.is_null() {
            panic!("Cannot create ONNX session options");
        };

        let mut session: *mut OrtSession = std::ptr::null_mut();
        let model_path = CString::new(model_path).unwrap();
        let session = unsafe {
            let _ = (*api).CreateSession.expect("CreateSession err")(
                env,
                model_path.as_ptr(),
                session_opts,
                &mut session,
            );
            session
        };
        if session.is_null() {
            panic!("Cannot create ONNX session");
        };

        ROR { api, env, session }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let _ror = ROR::new("models/mnist.onnx");
    }
}
