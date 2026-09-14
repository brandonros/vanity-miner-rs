use cust::module::{Module, ModuleJitOption};
use cust_raw::driver_sys;
use std::error::Error;
use std::ffi::{CStr, CString, c_void};
use std::os::raw::{c_char, c_uint};
use std::ptr;

pub(crate) fn load_module(ordinal: usize) -> Result<Module, Box<dyn Error + Send + Sync>> {
    println!("[{ordinal}] Loading module...");
    // An explicit CUBIN takes precedence over PTX_PATH and embedded PTX.
    // Surface loading failures instead of silently falling back.
    if let Some(cubin_path) = std::env::var_os("CUBIN_PATH") {
        let cubin_path = std::path::PathBuf::from(cubin_path);
        let module = Module::from_file(&cubin_path)
            .map_err(|e| format!("Failed to load CUBIN file {}: {}", cubin_path.display(), e))?;
        println!(
            "[{ordinal}] Module loaded from CUBIN: {}",
            cubin_path.display()
        );
        return Ok(module);
    }
    let ptx_owned;
    let ptx: &str = if let Ok(ptx_path) = std::env::var("PTX_PATH") {
        ptx_owned = std::fs::read_to_string(ptx_path)
            .map_err(|e| format!("Failed to read PTX file: {}", e))?;
        &ptx_owned
    } else {
        const EMBEDDED_PTX: &[u8] = include_bytes!(env!("KERNELS_PTX_PATH"));
        std::str::from_utf8(EMBEDDED_PTX)
            .map_err(|e| format!("Embedded PTX is not valid UTF-8: {}", e))?
    };
    let module = load_ptx_with_log(ordinal, ptx)?;
    println!("[{ordinal}] Module loaded");
    Ok(module)
}

fn load_ptx_with_log(ordinal: usize, ptx: &str) -> Result<Module, Box<dyn Error + Send + Sync>> {
    let cstr = CString::new(ptx).map_err(|e| format!("PTX contains nul bytes: {}", e))?;

    const LOG_CAP: usize = 16 * 1024;
    let mut info_log = vec![0u8; LOG_CAP];
    let mut error_log = vec![0u8; LOG_CAP];

    // Driver packs values directly into the *mut c_void slot when the payload fits.
    // LOG_VERBOSE = request detailed log
    // INFO/ERROR_LOG_BUFFER = pointer to buffer
    // *_LOG_BUFFER_SIZE_BYTES = capacity (in), bytes written (out)
    let mut options = [
        driver_sys::CUjit_option::CU_JIT_MAX_REGISTERS,
        driver_sys::CUjit_option::CU_JIT_LOG_VERBOSE,
        driver_sys::CUjit_option::CU_JIT_INFO_LOG_BUFFER,
        driver_sys::CUjit_option::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        driver_sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER,
        driver_sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    ];
    let mut option_values: [*mut c_void; 6] = [
        255usize as *mut c_void,
        1usize as *mut c_void,
        info_log.as_mut_ptr() as *mut c_void,
        LOG_CAP as *mut c_void,
        error_log.as_mut_ptr() as *mut c_void,
        LOG_CAP as *mut c_void,
    ];

    let mut module_ptr: driver_sys::CUmodule = ptr::null_mut();
    let res = unsafe {
        driver_sys::cuModuleLoadDataEx(
            &mut module_ptr,
            cstr.as_ptr() as *const c_void,
            options.len() as c_uint,
            options.as_mut_ptr(),
            option_values.as_mut_ptr(),
        )
    };

    let info_len = option_values[3] as usize;
    let error_len = option_values[5] as usize;
    let info_str = String::from_utf8_lossy(&info_log[..info_len.min(LOG_CAP)]);
    let error_str = String::from_utf8_lossy(&error_log[..error_len.min(LOG_CAP)]);

    if !info_str.trim().is_empty() {
        eprintln!("[{ordinal}] JIT info log ({info_len} bytes):\n{info_str}");
    }
    if !error_str.trim().is_empty() {
        eprintln!("[{ordinal}] JIT error log ({error_len} bytes):\n{error_str}");
    }
    eprintln!("[{ordinal}] cuModuleLoadDataEx raw result code: {:?}", res);

    if res != driver_sys::cudaError_enum::CUDA_SUCCESS {
        unsafe {
            let mut err_cstr: *const c_char = ptr::null();
            if driver_sys::cuGetErrorString(res, &mut err_cstr)
                == driver_sys::cudaError_enum::CUDA_SUCCESS
                && !err_cstr.is_null()
            {
                let msg = CStr::from_ptr(err_cstr).to_string_lossy();
                eprintln!("[{ordinal}] cuGetErrorString: {msg}");
            }
        }
        return Err(format!("cuModuleLoadDataEx failed: {:?}", res).into());
    }

    // The driver accepted the PTX; drop our raw handle and re-load via cust so the
    // caller gets a typed Module with cust's lifetime/drop machinery.
    let _ = unsafe { driver_sys::cuModuleUnload(module_ptr) };
    Module::from_ptx(ptx, &[ModuleJitOption::MaxRegisters(255)]).map_err(|e| e.into())
}
