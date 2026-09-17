//! Initialized, aligned, persistent device records, erased after their last use.
use cust::{
    memory::{CopyDestination, DeviceBuffer, DevicePointer},
    stream::Stream,
};
use logic::search::device_record::DeviceRecord;
use zeroize::{Zeroize, Zeroizing};

pub(crate) struct Records<'a, T: DeviceRecord> {
    bytes: DeviceBuffer<u8>,
    stream: &'a Stream,
    count: usize,
    marker: std::marker::PhantomData<T>,
}

impl<'a, T: DeviceRecord> Records<'a, T> {
    pub fn zeroed(count: usize, stream: &'a Stream) -> Result<Self, String> {
        let size = count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or("device allocation size overflow")?;
        if size == 0 {
            return Err("empty device record allocation".into());
        }
        Ok(Self {
            // CUDA allocations are aligned for all these integer-only records.
            bytes: DeviceBuffer::zeroed(size).map_err(|e| e.to_string())?,
            stream,
            count,
            marker: std::marker::PhantomData,
        })
    }

    pub fn from_slice(values: &[T], stream: &'a Stream) -> Result<Self, String> {
        let mut result = Self::zeroed(values.len(), stream)?;
        // SAFETY: DeviceRecord guarantees initialized bytes without padding.
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        result.bytes.copy_from(bytes).map_err(|e| e.to_string())?;
        Ok(result)
    }

    pub fn write(&mut self, values: &[T]) -> Result<(), String> {
        if values.len() != self.count {
            return Err("device record count changed".into());
        }
        // SAFETY: DeviceRecord guarantees initialized bytes without padding.
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        self.bytes.copy_from(bytes).map_err(|e| e.to_string())
    }

    pub fn pointer(&self) -> DevicePointer<u8> {
        // The driver transports addresses. T determines allocation size and
        // checked serialization; it need not implement cust's host-only trait.
        self.bytes.as_device_ptr()
    }

    pub fn clear_prefix(&mut self, count: usize) -> Result<(), String> {
        if count > self.count {
            return Err("device clear exceeds buffer".into());
        }
        self.bytes[..count * std::mem::size_of::<T>()]
            .set_8(0)
            .map_err(|e| e.to_string())
    }

    /// Ordered with subsequent work in this stream; no host buffer is borrowed.
    pub fn clear_async(&mut self) -> Result<(), String> {
        unsafe { self.bytes.set_8_async(0, self.stream) }.map_err(|e| e.to_string())
    }

    /// Caller synchronizes the last writing launch before reading.
    pub fn read(&self, count: usize) -> Result<Zeroizing<Vec<T>>, String>
    where
        T: Zeroize,
    {
        if count > self.count {
            return Err("device result exceeds buffer".into());
        }
        // SAFETY: DeviceRecord accepts every bit pattern, including all-zero.
        let zero = unsafe { std::mem::zeroed::<T>() };
        let mut result = Zeroizing::new(vec![zero; count]);
        let length = count * std::mem::size_of::<T>();
        if length != 0 {
            let bytes =
                unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr().cast::<u8>(), length) };
            self.bytes[..length]
                .copy_to(bytes)
                .map_err(|e| e.to_string())?;
        }
        Ok(result)
    }
}

impl<T: DeviceRecord> Drop for Records<'_, T> {
    fn drop(&mut self) {
        // Keep this guard on every error path, including partially queued stages.
        let result = self.stream.synchronize().and_then(|_| self.bytes.set_8(0));
        if result.is_err() {
            eprintln!("CUDA record erasure failed; releasing the device context.");
        }
    }
}

// Messages also live for the whole search and can contain private experiment data.
pub(crate) struct Message<'a> {
    pub bytes: DeviceBuffer<u8>,
    stream: &'a Stream,
}
impl<'a> Message<'a> {
    pub fn new(bytes: &[u8], stream: &'a Stream) -> Result<Self, String> {
        Ok(Self {
            bytes: DeviceBuffer::from_slice(if bytes.is_empty() { &[0] } else { bytes })
                .map_err(|e| e.to_string())?,
            stream,
        })
    }
}
impl Drop for Message<'_> {
    fn drop(&mut self) {
        if self
            .stream
            .synchronize()
            .and_then(|_| self.bytes.set_8(0))
            .is_err()
        {
            eprintln!("CUDA message erasure failed; releasing the device context.");
        }
    }
}
