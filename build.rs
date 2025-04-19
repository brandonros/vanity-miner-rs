use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("path/to/gpu/crate/root")
        .copy_to("some/path.ptx")
        .build()
        .unwrap();
}
