use cust::device::Device;
use cust::module::{Module, ModuleJitOption};
use cust::prelude::Context;
use cust::stream::{Stream, StreamFlags};
use cust::util::SliceExt;
use cust::memory::CopyDestination;
use cust::{launch, CudaFlags};
use rand::Rng;
use std::error::Error;
use std::sync::{Arc, RwLock};

use common::GlobalStats;

pub fn device_main_ethereum_vanity(
    ordinal: usize, 
    vanity_prefix: String, 
    vanity_suffix: String,
    module: &Module,
    global_stats: Arc<GlobalStats>
) -> Result<(), Box<dyn Error + Send + Sync>> {
    todo!()
}
