use inkwell::context::Context;
use inkwell::data_layout::DataLayout;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::module::Module;
use inkwell::targets::TargetTriple;
use inkwell::values::BasicMetadataValueEnum;
use std::fs;

fn add_nvvm_ir_version<'ctx, 'module>(context: &'ctx Context, module: &'module Module<'ctx>) {
    // Add NVVM IR version metadata: !nvvm.ir.version = !{!0}
    // !0 = !{i32 2, i32 0}
    let i32_type = context.i32_type();
    
    // Create constant values for version 2.0
    let major_version = i32_type.const_int(2, false);
    let minor_version = i32_type.const_int(0, false);
    
    // Create metadata node directly with the constant values
    let version_metadata = context.metadata_node(&[
        BasicMetadataValueEnum::IntValue(major_version),
        BasicMetadataValueEnum::IntValue(minor_version),
    ]);
    
    // Add to named metadata
    module.add_global_metadata("nvvmir.version", &version_metadata).unwrap();
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <filename>", args[0]);
        std::process::exit(1);
    }
    
    // create context
    let context = Context::create();

    // load riscv64gc-unknown-none-elf llvm ir
    let filename = args[1].clone();
    let risc_ir = fs::read(filename)?;
    let risc_memory_buffer = MemoryBuffer::create_from_memory_range(&risc_ir, "risc_ir");
    let risc_module = context.create_module_from_ir(risc_memory_buffer)?;

    // jumpstart ptx module from risc module
    let ptx_module = risc_module.clone();

    // Set target triple
    let target_triple = "nvptx64-nvidia-cuda";
    ptx_module.set_triple(&TargetTriple::create(target_triple));

    // Set data layout
    let data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8";
    ptx_module.set_data_layout(&DataLayout::create(data_layout));

    // Add NVVM IR version metadata
    add_nvvm_ir_version(&context, &ptx_module);

    // Write to .ll file
    let llvm_ir = ptx_module.print_to_string().to_string();
    fs::write("/tmp/output.ll", llvm_ir).expect("Unable to write file");
    println!("LLVM IR written to output.ll");

    // Write to .bc file
    assert_eq!(ptx_module.write_bitcode_to_path("/tmp/output.bc"), true);
    println!("LLVM bitcode written to output.bc");
    
    Ok(())
}
