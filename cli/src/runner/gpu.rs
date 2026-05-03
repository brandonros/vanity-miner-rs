use crate::args::Command;
use crate::common::{GlobalStats, SharedBestHash};
use crate::modes;
use crate::runner::Runner;
use backtrace::Backtrace;
use cust::device::Device;
use cust::module::{Module, ModuleJitOption};
use cust::prelude::Context;
use cust::CudaFlags;
use cust_raw::driver_sys;
use std::error::Error;
use std::ffi::{CStr, CString, c_void};
use std::os::raw::{c_char, c_uint};
use std::ptr;
use std::sync::{Arc, RwLock};

pub struct GpuRunner {
    num_devices: usize,
}

impl GpuRunner {
    pub fn new() -> Result<Self, Box<dyn Error + Send + Sync>> {
        cust::init(CudaFlags::empty())?;
        let num_devices = Device::num_devices()? as usize;
        println!("Found {} CUDA devices", num_devices);
        Ok(Self { num_devices })
    }

    fn load_module(ordinal: usize) -> Result<Module, Box<dyn Error + Send + Sync>> {
        let device = Device::get_device(ordinal as u32)?;
        let ctx = Context::new(device)?;
        cust::context::CurrentContext::set_current(&ctx)?;

        println!("[{ordinal}] Loading module...");
        let module = if let Ok(cubin_path) = std::env::var("CUBIN_PATH") {
            let cubin = std::fs::read(cubin_path)
                .map_err(|e| format!("Failed to read CUBIN file: {}", e))?;
            Module::from_cubin(cubin, &[])?
        } else if let Ok(ptx_path) = std::env::var("PTX_PATH") {
            let ptx = std::fs::read_to_string(ptx_path)
                .map_err(|e| format!("Failed to read PTX file: {}", e))?;
            Self::load_ptx_with_log(ordinal, &ptx)?
        } else {
            const EMBEDDED_PTX: &[u8] = include_bytes!(env!("KERNELS_PTX_PATH"));
            let ptx = std::str::from_utf8(EMBEDDED_PTX)
                .map_err(|e| format!("Embedded PTX is not valid UTF-8: {}", e))?;
            Self::load_ptx_with_log(ordinal, ptx)?
        };
        println!("[{ordinal}] Module loaded");
        Ok(module)
    }

    /// JIT-compile PTX via the raw driver API with CU_JIT_{INFO,ERROR}_LOG_BUFFER
    /// so we can surface the driver's actual diagnostic when JIT compilation fails.
    /// cust's ModuleJitOption doesn't expose the log buffers, so on UnknownError
    /// the only way to see "register X exceeded" / "unsupported sm_XX" / etc. is
    /// to round-trip through cuModuleLoadDataEx ourselves.
    fn load_ptx_with_log(ordinal: usize, ptx: &str) -> Result<Module, Box<dyn Error + Send + Sync>> {
        let cstr = CString::new(ptx).map_err(|e| format!("PTX contains nul bytes: {}", e))?;

        const LOG_CAP: usize = 16 * 1024;
        let mut info_log = vec![0u8; LOG_CAP];
        let mut error_log = vec![0u8; LOG_CAP];

        // Driver packs scalar values directly into the *mut c_void slot when they fit.
        // Order must line up with `options` below.
        let mut options = [
            driver_sys::CUjit_option::CU_JIT_MAX_REGISTERS,
            driver_sys::CUjit_option::CU_JIT_LOG_VERBOSE,
            driver_sys::CUjit_option::CU_JIT_INFO_LOG_BUFFER,
            driver_sys::CUjit_option::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            driver_sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER,
            driver_sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        ];
        let mut option_values: [*mut c_void; 6] = [
            256usize as *mut c_void,
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

        if res != driver_sys::cudaError_enum::CUDA_SUCCESS {
            let driver_msg = unsafe {
                let mut err_cstr: *const c_char = ptr::null();
                if driver_sys::cuGetErrorString(res, &mut err_cstr)
                    == driver_sys::cudaError_enum::CUDA_SUCCESS
                    && !err_cstr.is_null()
                {
                    CStr::from_ptr(err_cstr).to_string_lossy().into_owned()
                } else {
                    format!("{:?}", res)
                }
            };
            return Err(format!(
                "cuModuleLoadDataEx failed on device {ordinal}: {driver_msg} ({:?}); error log: {}",
                res,
                error_str.trim()
            )
            .into());
        }

        // Driver accepted the PTX. Drop the raw handle and re-load through cust so
        // the rest of the runner gets a typed Module with cust's drop semantics.
        let _ = unsafe { driver_sys::cuModuleUnload(module_ptr) };
        Module::from_ptx(ptx, &[ModuleJitOption::MaxRegisters(256)]).map_err(|e| e.into())
    }
}

impl Runner for GpuRunner {
    fn device_count(&self) -> usize {
        self.num_devices
    }

    fn run(&self, command: &Command, stats: Arc<GlobalStats>) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Set up panic hook for better error reporting
        std::panic::set_hook(Box::new(|panic_info| {
            let backtrace = Backtrace::new();
            eprintln!("Thread panicked: {}", panic_info);
            eprintln!("Backtrace:\n{:?}", backtrace);
        }));

        // Create shared state for shallenge mode
        let shared_best_hash: Option<Arc<RwLock<SharedBestHash>>> = match command {
            Command::Shallenge { target_hash, .. } => {
                let target_hash_bytes = hex::decode(target_hash)?;
                let mut initial_target = [0u8; 32];
                initial_target.copy_from_slice(&target_hash_bytes);
                Some(Arc::new(RwLock::new(SharedBestHash::new(initial_target))))
            }
            _ => None,
        };

        // Spawn device threads
        let mut handles = Vec::new();
        for i in 0..self.num_devices {
            println!("Starting device {}", i);
            let command_clone = command.clone();
            let shared_best_hash_clone = shared_best_hash.clone();
            let stats_clone = Arc::clone(&stats);

            handles.push(std::thread::spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
                let module = Self::load_module(i)?;

                match command_clone {
                    Command::SolanaVanity { prefix, suffix } => {
                        modes::solana::gpu::run(i, prefix, suffix, &module, stats_clone)
                    }
                    Command::BitcoinVanity { prefix, suffix } => {
                        modes::bitcoin::gpu::run(i, prefix, suffix, &module, stats_clone)
                    }
                    Command::EthereumVanity { prefix, suffix } => {
                        modes::ethereum::gpu::run(i, prefix, suffix, &module, stats_clone)
                    }
                    Command::Shallenge { username, .. } => {
                        let shared = shared_best_hash_clone.expect("SharedBestHash required for shallenge mode");
                        modes::shallenge::gpu::run(i, username, shared, &module, stats_clone)
                    }
                }.map_err(|e| {
                    let bt = Backtrace::new();
                    eprintln!("Error in device {}: {}", i, e);
                    eprintln!("Backtrace:\n{:?}", bt);
                    e
                })
            }));
        }

        // Wait for threads
        for (i, handle) in handles.into_iter().enumerate() {
            match handle.join() {
                Ok(result) => {
                    if let Err(e) = result {
                        eprintln!("Device {} returned error: {}", i, e);
                        return Err(e);
                    }
                }
                Err(panic_payload) => {
                    eprintln!("Device {} thread panicked!", i);
                    let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "Unknown panic".to_string()
                    };
                    eprintln!("Panic message: {}", panic_msg);
                    return Err(format!("Device {} panicked: {}", i, panic_msg).into());
                }
            }
        }

        Ok(())
    }
}
