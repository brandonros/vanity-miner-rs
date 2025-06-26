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

    if args.len() != 4 {
        eprintln!("Usage: {} <mode> <riscv_filename> <ptx_filename>", args[0]);
        std::process::exit(1);
    }
    let mode = args[1].clone();
    let risc_filename = args[2].clone();
    let out_filename = args[3].clone();
    
    // create context
    let context = Context::create();

    // load riscv64gc-unknown-none-elf llvm ir
    let risc_ir = fs::read(risc_filename)?;
    let risc_memory_buffer = MemoryBuffer::create_from_memory_range(&risc_ir, "risc_ir");
    let risc_module = context.create_module_from_ir(risc_memory_buffer)?;

    // jumpstart ptx module from risc module
    let out_module = risc_module.clone();

    if mode == "nvptx64" {
        // Set target triple
        let target_triple = "nvptx64-nvidia-cuda";
        out_module.set_triple(&TargetTriple::create(target_triple));

        // Set data layout
        let data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8";
        out_module.set_data_layout(&DataLayout::create(data_layout));

        // Add NVVM IR version metadata
        add_nvvm_ir_version(&context, &out_module);
    } else if mode == "spirv64" {
        // Set target triple
        let target_triple = "spirv64-unknown-unknown";
        out_module.set_triple(&TargetTriple::create(target_triple));

        // Set data layout
        let data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024";
        out_module.set_data_layout(&DataLayout::create(data_layout));
    } else {
        eprintln!("Invalid mode: {}", mode);
        std::process::exit(1);
    }

    // Write to .ll file
    let llvm_ir = out_module.print_to_string().to_string();
    fs::write(&out_filename, llvm_ir).expect("Unable to write file");
    println!("LLVM IR written to {}", out_filename);
    
    Ok(())
}
