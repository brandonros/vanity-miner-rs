#include <stdio.h>
#include <stdlib.h>
#include <nvvm.h>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.bc> <libintrinsics.bc>\n", argv[0]);
        return 1;
    }
    
    // Read your Rust-generated bitcode
    fprintf(stderr, "Reading bitcode from %s\n", argv[1]);
    FILE* bitcode_file = fopen(argv[1], "rb");
    if (!bitcode_file) {
        fprintf(stderr, "Error: Could not open %s\n", argv[1]);
        return 1;
    }
    
    fseek(bitcode_file, 0, SEEK_END);
    size_t bitcode_size = ftell(bitcode_file);
    rewind(bitcode_file);
    char* bitcode = malloc(bitcode_size);
    fread(bitcode, 1, bitcode_size, bitcode_file);
    fclose(bitcode_file);

    // Read libintrinsics
    fprintf(stderr, "Reading libintrinsics from %s\n", argv[2]);
    FILE* libintrinsics_file = fopen(argv[2], "rb");
    if (!libintrinsics_file) {
        fprintf(stderr, "Error: Could not open %s\n", argv[2]);
        return 1;
    }

    fseek(libintrinsics_file, 0, SEEK_END);
    size_t libintrinsics_size = ftell(libintrinsics_file);
    rewind(libintrinsics_file);
    char* libintrinsics = malloc(libintrinsics_size);
    fread(libintrinsics, 1, libintrinsics_size, libintrinsics_file);
    fclose(libintrinsics_file);

    // Create NVVM program
    fprintf(stderr, "Creating NVVM program\n");
    nvvmProgram prog;
    nvvmResult result = nvvmCreateProgram(&prog);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "Error creating NVVM program: %d\n", result);
        free(bitcode);
        free(libintrinsics);
        return 1;
    }

    // Add your LLVM bitcode module
    fprintf(stderr, "Adding module to program\n");
    result = nvvmAddModuleToProgram(prog, bitcode, bitcode_size, "mymodule");
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "Error adding module to program: %d\n", result);
        
        // Get compilation log for debugging
        size_t log_size;
        nvvmGetProgramLogSize(prog, &log_size);
        if (log_size > 1) {
            char* log = malloc(log_size);
            nvvmGetProgramLog(prog, log);
            fprintf(stderr, "NVVM Log:\n%s\n", log);
            free(log);
        }
        
        nvvmDestroyProgram(&prog);
        free(bitcode);
        free(libintrinsics);
        return 1;
    }

    // Add libintrinsics
    fprintf(stderr, "Adding libintrinsics to program\n");
    result = nvvmLazyAddModuleToProgram(prog, libintrinsics, libintrinsics_size, "libintrinsics");
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "Error adding module to program: %d\n", result);
        
        // Get compilation log for debugging
        size_t log_size;
        nvvmGetProgramLogSize(prog, &log_size);
        if (log_size > 1) {
            char* log = malloc(log_size);
            nvvmGetProgramLog(prog, log);
            fprintf(stderr, "NVVM Log:\n%s\n", log);
            free(log);
        }
        
        nvvmDestroyProgram(&prog);
        free(bitcode);
        free(libintrinsics);
        return 1;
    }

    // Verify the program before compilation
    fprintf(stderr, "Verifying program\n");
    const char* options[] = {"-arch=compute_120"};
    result = nvvmVerifyProgram(prog, 1, options);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "Error verifying program: %d\n", result);
        
        // Get verification log for debugging
        size_t log_size;
        nvvmGetProgramLogSize(prog, &log_size);
        if (log_size > 1) {
            char* log = malloc(log_size);
            nvvmGetProgramLog(prog, log);
            fprintf(stderr, "NVVM Verification Log:\n%s\n", log);
            free(log);
        }
        
        nvvmDestroyProgram(&prog);
        free(bitcode);
        free(libintrinsics);
        return 1;
    }

    // Compile to PTX
    fprintf(stderr, "Compiling program\n");
    result = nvvmCompileProgram(prog, 1, options);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "Error compiling program: %d\n", result);
        
        // Get compilation log for debugging
        size_t log_size;
        nvvmGetProgramLogSize(prog, &log_size);
        if (log_size > 1) {
            char* log = malloc(log_size);
            nvvmGetProgramLog(prog, log);
            fprintf(stderr, "NVVM Log:\n%s\n", log);
            free(log);
        }
        
        nvvmDestroyProgram(&prog);
        free(bitcode);
        free(libintrinsics);
        return 1;
    }

    // Get PTX result
    fprintf(stderr, "Getting compiled result size\n");
    size_t ptx_size;
    result = nvvmGetCompiledResultSize(prog, &ptx_size);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "Error getting compiled result size: %d\n", result);
        nvvmDestroyProgram(&prog);
        free(bitcode);
        free(libintrinsics);
        return 1;
    }

    fprintf(stderr, "Getting compiled result\n");
    char* ptx = malloc(ptx_size);
    result = nvvmGetCompiledResult(prog, ptx);
    if (result != NVVM_SUCCESS) {
        fprintf(stderr, "Error getting compiled result: %d\n", result);
        nvvmDestroyProgram(&prog);
        free(bitcode);
        free(ptx);
        return 1;
    }

    // Write PTX to stdout
    fprintf(stderr, "Writing PTX to stdout\n");
    fwrite(ptx, 1, ptx_size - 1, stdout); // -1 to exclude null terminator

    // Optional: Write stats to stderr so they don't interfere with PTX output
    fprintf(stderr, "Successfully compiled LLVM bitcode to PTX!\n");
    fprintf(stderr, "Input: %s (%zu bytes)\n", argv[1], bitcode_size);
    fprintf(stderr, "Output: PTX (%zu bytes)\n", ptx_size - 1);

    // Clean up
    nvvmDestroyProgram(&prog);
    free(bitcode);
    free(libintrinsics);
    free(ptx);
    return 0;
}
