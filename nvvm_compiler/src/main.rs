use std::env;
use std::fs::File;
use std::io::{self, Read, Write};
use std::ffi::{CString, CStr};
use std::ptr;
use std::os::raw::{c_char, c_int, c_void};

// NVVM C bindings
#[repr(C)]
#[derive(Copy, Clone)]
pub struct nvvmProgram(*mut c_void);

#[repr(C)]
#[derive(Debug, PartialEq)]
pub enum nvvmResult {
    NVVM_SUCCESS = 0,
    NVVM_ERROR_OUT_OF_MEMORY = 1,
    NVVM_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVVM_ERROR_IR_VERSION_MISMATCH = 3,
    NVVM_ERROR_INVALID_INPUT = 4,
    NVVM_ERROR_INVALID_PROGRAM = 5,
    NVVM_ERROR_INVALID_IR = 6,
    NVVM_ERROR_INVALID_OPTION = 7,
    NVVM_ERROR_NO_MODULE_IN_PROGRAM = 8,
    NVVM_ERROR_COMPILATION = 9,
}

#[link(name = "nvvm")]
unsafe extern "C" {
    fn nvvmCreateProgram(prog: *mut nvvmProgram) -> nvvmResult;
    fn nvvmDestroyProgram(prog: *mut nvvmProgram) -> nvvmResult;
    fn nvvmAddModuleToProgram(
        prog: nvvmProgram,
        buffer: *const c_char,
        size: usize,
        name: *const c_char,
    ) -> nvvmResult;
    fn nvvmLazyAddModuleToProgram(
        prog: nvvmProgram,
        buffer: *const c_char,
        size: usize,
        name: *const c_char,
    ) -> nvvmResult;
    fn nvvmVerifyProgram(
        prog: nvvmProgram,
        num_options: c_int,
        options: *const *const c_char,
    ) -> nvvmResult;
    fn nvvmCompileProgram(
        prog: nvvmProgram,
        num_options: c_int,
        options: *const *const c_char,
    ) -> nvvmResult;
    fn nvvmGetCompiledResultSize(prog: nvvmProgram, size: *mut usize) -> nvvmResult;
    fn nvvmGetCompiledResult(prog: nvvmProgram, buffer: *mut c_char) -> nvvmResult;
    fn nvvmGetProgramLogSize(prog: nvvmProgram, size: *mut usize) -> nvvmResult;
    fn nvvmGetProgramLog(prog: nvvmProgram, buffer: *mut c_char) -> nvvmResult;
}

fn read_file(path: &str) -> Result<Vec<u8>, io::Error> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

fn get_nvvm_log(prog: nvvmProgram) -> Option<String> {
    unsafe {
        let mut log_size = 0;
        if nvvmGetProgramLogSize(prog, &mut log_size) != nvvmResult::NVVM_SUCCESS || log_size <= 1 {
            return None;
        }
        
        let mut log_buffer = vec![0u8; log_size];
        if nvvmGetProgramLog(prog, log_buffer.as_mut_ptr() as *mut c_char) != nvvmResult::NVVM_SUCCESS {
            return None;
        }
        
        // Convert to string, removing null terminator
        log_buffer.pop();
        String::from_utf8(log_buffer).ok()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <input.bc> <libintrinsics.bc> <arch>", args[0]);
        eprintln!("Example: {} input.bc libintrinsics.bc compute_75", args[0]);
        return Err("Invalid arguments".into());
    }

    // Read bitcode files
    eprintln!("Reading bitcode from {}", args[1]);
    let bitcode = read_file(&args[1])
        .map_err(|e| format!("Error: Could not open {}: {}", args[1], e))?;

    eprintln!("Reading libintrinsics from {}", args[2]);
    let libintrinsics = read_file(&args[2])
        .map_err(|e| format!("Error: Could not open {}: {}", args[2], e))?;

    // Prepare architecture option
    let arch_option = format!("-arch={}", args[3]);
    let arch_option_cstr = CString::new(arch_option)?;
    let opt_cstr = CString::new("-opt=3")?;
    
    let options: Vec<*const c_char> = vec![
        arch_option_cstr.as_ptr(),
        opt_cstr.as_ptr(),
    ];

    unsafe {
        // Create NVVM program
        eprintln!("Creating NVVM program");
        let mut prog = nvvmProgram(ptr::null_mut());
        let mut result = nvvmCreateProgram(&mut prog);
        if result != nvvmResult::NVVM_SUCCESS {
            return Err(format!("Error creating NVVM program: {:?}", result).into());
        }

        // Add main module
        eprintln!("Adding module to program");
        let module_name = CString::new("mymodule")?;
        result = nvvmAddModuleToProgram(
            prog,
            bitcode.as_ptr() as *const c_char,
            bitcode.len(),
            module_name.as_ptr(),
        );
        if result != nvvmResult::NVVM_SUCCESS {
            if let Some(log) = get_nvvm_log(prog) {
                eprintln!("NVVM Log:\n{}", log);
            }
            nvvmDestroyProgram(&mut prog);
            return Err(format!("Error adding module to program: {:?}", result).into());
        }

        // Add libintrinsics
        eprintln!("Adding libintrinsics to program");
        let libintrinsics_name = CString::new("libintrinsics")?;
        result = nvvmLazyAddModuleToProgram(
            prog,
            libintrinsics.as_ptr() as *const c_char,
            libintrinsics.len(),
            libintrinsics_name.as_ptr(),
        );
        if result != nvvmResult::NVVM_SUCCESS {
            if let Some(log) = get_nvvm_log(prog) {
                eprintln!("NVVM Log:\n{}", log);
            }
            nvvmDestroyProgram(&mut prog);
            return Err(format!("Error adding libintrinsics to program: {:?}", result).into());
        }

        // Verify program
        eprintln!("Verifying program with arch: {}", args[3]);
        result = nvvmVerifyProgram(prog, options.len() as c_int, options.as_ptr());
        if result != nvvmResult::NVVM_SUCCESS {
            if let Some(log) = get_nvvm_log(prog) {
                eprintln!("NVVM Verification Log:\n{}", log);
            }
            nvvmDestroyProgram(&mut prog);
            return Err(format!("Error verifying program: {:?}", result).into());
        }

        // Compile to PTX
        eprintln!("Compiling program with arch: {}", args[3]);
        result = nvvmCompileProgram(prog, options.len() as c_int, options.as_ptr());
        if result != nvvmResult::NVVM_SUCCESS {
            if let Some(log) = get_nvvm_log(prog) {
                eprintln!("NVVM Log:\n{}", log);
            }
            nvvmDestroyProgram(&mut prog);
            return Err(format!("Error compiling program: {:?}", result).into());
        }

        // Get PTX result size
        eprintln!("Getting compiled result size");
        let mut ptx_size = 0;
        result = nvvmGetCompiledResultSize(prog, &mut ptx_size);
        if result != nvvmResult::NVVM_SUCCESS {
            nvvmDestroyProgram(&mut prog);
            return Err(format!("Error getting compiled result size: {:?}", result).into());
        }

        // Get PTX result
        eprintln!("Getting compiled result");
        let mut ptx = vec![0u8; ptx_size];
        result = nvvmGetCompiledResult(prog, ptx.as_mut_ptr() as *mut c_char);
        if result != nvvmResult::NVVM_SUCCESS {
            nvvmDestroyProgram(&mut prog);
            return Err(format!("Error getting compiled result: {:?}", result).into());
        }

        // Write PTX to stdout (excluding null terminator)
        eprintln!("Writing PTX to stdout");
        if ptx_size > 0 {
            io::stdout().write_all(&ptx[..ptx_size - 1])?;
        }

        // Print stats to stderr
        eprintln!("Successfully compiled LLVM bitcode to PTX!");
        eprintln!("Input: {} ({} bytes)", args[1], bitcode.len());
        eprintln!("Architecture: {}", args[3]);
        eprintln!("Output: PTX ({} bytes)", ptx_size.saturating_sub(1));

        // Clean up
        nvvmDestroyProgram(&mut prog);
    }

    Ok(())
}