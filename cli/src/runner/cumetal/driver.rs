//! Small dynamically loaded CUDA Driver API subset used by the CuMetal host.
//! Signatures match CuMetal's public CUDA driver header; no NVIDIA SDK is needed.
use super::Error;
use libloading::Library;
use std::{
    ffi::{CString, c_void},
    path::Path,
    ptr,
    rc::Rc,
};
type Handle = *mut c_void;
type Init = unsafe extern "C" fn(u32) -> i32;
type Create = unsafe extern "C" fn(*mut Handle, u32, i32) -> i32;
type Destroy = unsafe extern "C" fn(Handle) -> i32;
type Alloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
type Free = unsafe extern "C" fn(u64) -> i32;
type ToDevice = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type ToHost = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
type Load = unsafe extern "C" fn(*mut Handle, *const i8) -> i32;
type Function = unsafe extern "C" fn(*mut Handle, Handle, *const i8) -> i32;
type Launch = unsafe extern "C" fn(
    Handle,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    Handle,
    *mut Handle,
    *mut Handle,
) -> i32;
type Sync = unsafe extern "C" fn() -> i32;
fn check(code: i32, name: &str) -> Result<(), Error> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("{name} failed: CUDA error {code}").into())
    }
}
pub struct Driver {
    context: Handle,
    destroy: Destroy,
    alloc: Alloc,
    free: Free,
    to_device: ToDevice,
    to_host: ToHost,
    load: Load,
    unload: Destroy,
    function: Function,
    launch: Launch,
    sync: Sync,
    _library: Library,
}
impl Driver {
    pub fn open(path: &Path) -> Result<Rc<Self>, Error> {
        // SAFETY: symbols are loaded with the documented driver ABI. Library
        // remains alive until all owning buffers/modules have been released.
        unsafe {
            let library = Library::new(path)?;
            let init: Init = *library.get(b"cuInit\0")?;
            let create: Create = *library.get(b"cuCtxCreate\0")?;
            let mut driver = Self {
                context: ptr::null_mut(),
                destroy: *library.get(b"cuCtxDestroy\0")?,
                alloc: *library.get(b"cuMemAlloc\0")?,
                free: *library.get(b"cuMemFree\0")?,
                to_device: *library.get(b"cuMemcpyHtoD\0")?,
                to_host: *library.get(b"cuMemcpyDtoH\0")?,
                load: *library.get(b"cuModuleLoad\0")?,
                unload: *library.get(b"cuModuleUnload\0")?,
                function: *library.get(b"cuModuleGetFunction\0")?,
                launch: *library.get(b"cuLaunchKernel\0")?,
                sync: *library.get(b"cuCtxSynchronize\0")?,
                _library: library,
            };
            check(init(0), "cuInit")?;
            check(create(&mut driver.context, 0, 0), "cuCtxCreate")?;
            Ok(Rc::new(driver))
        }
    }
    pub fn buffer(self: &Rc<Self>, initial: &[u8]) -> Result<Buffer, Error> {
        // Guard both sides. Empty inputs still get a valid device address.
        let mut bytes = vec![0xa5; initial.len() + 32];
        bytes[16..16 + initial.len()].copy_from_slice(initial);
        let mut b = Buffer {
            driver: self.clone(),
            base: 0,
            size: initial.len(),
        };
        unsafe {
            check((self.alloc)(&mut b.base, bytes.len()), "cuMemAlloc")?;
            check(
                (self.to_device)(b.base, bytes.as_ptr().cast(), bytes.len()),
                "cuMemcpyHtoD",
            )?;
        }
        Ok(b)
    }
    pub fn module(self: &Rc<Self>, path: &Path, entry: &str) -> Result<Module, Error> {
        let path = CString::new(path.as_os_str().as_encoded_bytes())?;
        let name = CString::new(entry)?;
        let mut m = Module {
            driver: self.clone(),
            handle: ptr::null_mut(),
            function: ptr::null_mut(),
            temporary: None,
        };
        unsafe {
            check((self.load)(&mut m.handle, path.as_ptr()), "cuModuleLoad")?;
            check(
                (self.function)(&mut m.function, m.handle, name.as_ptr()),
                "cuModuleGetFunction",
            )?;
        }
        Ok(m)
    }
}
impl Drop for Driver {
    fn drop(&mut self) {
        if !self.context.is_null() {
            unsafe {
                (self.destroy)(self.context);
            }
        }
    }
}
pub struct Buffer {
    driver: Rc<Driver>,
    base: u64,
    size: usize,
}
impl Buffer {
    #[cfg(feature = "crypto-cli")]
    pub fn clear(&self) -> Result<(), Error> {
        let zeros = vec![0u8; self.size + 32];
        unsafe {
            check((self.driver.sync)(), "cuCtxSynchronize")?;
            check(
                (self.driver.to_device)(self.base, zeros.as_ptr().cast(), zeros.len()),
                "cuMemcpyHtoD",
            )
        }
    }

    pub fn pointer(&self) -> u64 {
        self.base + 16
    }
    pub fn read(&self) -> Result<Vec<u8>, Error> {
        let mut bytes = vec![0; self.size + 32];
        unsafe {
            check(
                (self.driver.to_host)(bytes.as_mut_ptr().cast(), self.base, bytes.len()),
                "cuMemcpyDtoH",
            )?;
        }
        if bytes[..16]
            .iter()
            .chain(bytes[self.size + 16..].iter())
            .any(|b| *b != 0xa5)
        {
            return Err("GPU buffer guard overwritten".into());
        }
        Ok(bytes[16..self.size + 16].to_vec())
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        if self.base != 0 {
            unsafe {
                (self.driver.free)(self.base);
            }
        }
    }
}
pub struct Module {
    pub(super) temporary: Option<super::module::TemporaryDirectory>,
    driver: Rc<Driver>,
    handle: Handle,
    function: Handle,
}
impl Module {
    pub fn launch(&self, values: &mut [u64], blocks: u32, threads: u32) -> Result<(), Error> {
        let mut args: Vec<Handle> = values.iter_mut().map(|v| (v as *mut u64).cast()).collect();
        args.push(ptr::null_mut());
        unsafe {
            check(
                (self.driver.launch)(
                    self.function,
                    blocks,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?;
            check((self.driver.sync)(), "cuCtxSynchronize")?;
        }
        Ok(())
    }
}
impl Drop for Module {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                (self.driver.unload)(self.handle);
            }
        }
    }
}
