mod add;

use cust::device::Device;
use cust::module::{Module, ModuleJitOption};
use cust::prelude::Context;
use cust::CudaFlags;
use std::error::Error;
use std::sync::{Arc, RwLock};

fn device_main(
    ordinal: usize,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let device = Device::get_device(ordinal as u32)?;
    let ctx = Context::new(device)?;
    cust::context::CurrentContext::set_current(&ctx)?;

    println!("[{ordinal}] Loading module...");
    let cubin_path = std::env::var("CUBIN_PATH")
        .map_err(|_| "CUBIN_PATH environment variable is required")?;
    let cubin = std::fs::read(cubin_path)
        .map_err(|e| format!("Failed to read CUBIN file: {}", e))?;
    let module = Module::from_cubin(cubin, &[
        ModuleJitOption::MaxRegisters(256),
    ])?;
    println!("[{ordinal}] Module loaded");

    add::device_main_add(ordinal, &module)
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    cust::init(CudaFlags::empty())?;
    let num_devices = Device::num_devices()?;
    println!("Found {} CUDA devices", num_devices);

    let mut handles = Vec::new();
    for i in 0..num_devices as usize {
        println!("Starting device {}", i);
        handles.push(std::thread::spawn(move || device_main(
            i,
        )));
    }

    for handle in handles {
        handle.join().unwrap().unwrap();
    }

    Ok(())
}