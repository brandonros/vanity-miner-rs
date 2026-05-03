{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { nixpkgs, rust-overlay, ... }:
    let
      system = "aarch64-linux";
      # allowUnfree is required because CUDA is unfree.
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        overlays = [ rust-overlay.overlays.default ];
      };
      lib = pkgs.lib;

      # ---- CUDA toolkit (Nix-managed) ----
      # CUDA 13.2 → NVVM 22.0 → PTX 9.2 → needs driver 580.x+ (CUDA 13) at runtime.
      # `cudatoolkit` is the kitchen-sink symlinkJoin maintained by nixpkgs —
      # every header path and lib layout is already wired correctly. The host
      # NVIDIA driver (libcuda.so.1) is needed at runtime; it is *not* shimmed
      # in here — supply it via the system or extend LD_LIBRARY_PATH yourself
      # before running CUDA programs.
      cudaRoot = pkgs.cudaPackages_13_2.cudatoolkit;

      # Pin matches rust-toolchain.toml. We use override (not
      # fromRustupToolchainFile) because we need to add the riscv64 target
      # for the kernels crate plus rust-analyzer/rust-src extensions for IDE
      # support — fromRustupToolchainFile doesn't compose with overrides.
      toolchain = pkgs.rust-bin.stable."1.86.0".default.override {
        extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
        targets = [ "riscv64gc-unknown-none-elf" ];
      };

      # ---- LLVM 19 ----
      llvm19 = pkgs.llvmPackages_19;
      llvm19Bin = lib.getBin llvm19.llvm;
      llvm19Dev = lib.getDev llvm19.llvm;
      llvm19CompatTools = pkgs.symlinkJoin {
        name = "llvm19-compat-tools";
        paths = [
          (pkgs.writeShellScriptBin "opt-19" ''exec ${llvm19Bin}/bin/opt "$@"'')
          (pkgs.writeShellScriptBin "llvm-as-19" ''exec ${llvm19Bin}/bin/llvm-as "$@"'')
          (pkgs.writeShellScriptBin "llvm-dis-19" ''exec ${llvm19Bin}/bin/llvm-dis "$@"'')
          (pkgs.writeShellScriptBin "llc-19" ''exec ${llvm19Bin}/bin/llc "$@"'')
        ];
      };

      devShell = pkgs.mkShell {
        CUDA_HOME = "${cudaRoot}";
        CUDA_ROOT = "${cudaRoot}";
        CUDA_PATH = "${cudaRoot}";
        CUDA_TOOLKIT_ROOT_DIR = "${cudaRoot}";
        # Cover both lib/ (nix-style) and lib64/ (FHS-style) so downstream
        # build.rs scripts that probe either layout resolve libcudart + stubs.
        CUDA_LIBRARY_PATH =
          "${cudaRoot}/lib:${cudaRoot}/lib64:${cudaRoot}/lib/stubs:${cudaRoot}/lib64/stubs";
        LLVM_CONFIG_19 = "${llvm19Dev}/bin/llvm-config";
        LIBCLANG_PATH = "${lib.getLib llvm19.libclang}/lib";

        nativeBuildInputs = [
          toolchain
          pkgs.gcc
          pkgs.pkg-config
          pkgs.cmake
          pkgs.ninja
          cudaRoot
          llvm19.clang
          llvm19.libclang
          llvm19Bin
          llvm19Dev
          llvm19CompatTools
        ];
        buildInputs = [
          pkgs.openssl
          pkgs.libxml2
          pkgs.zlib
          pkgs.ncurses
          pkgs.stdenv.cc.cc.lib
        ];

        shellHook = ''
          export PATH="${llvm19CompatTools}/bin:${llvm19Bin}/bin:${llvm19Dev}/bin:${cudaRoot}/bin:${cudaRoot}/nvvm/bin:$PATH"
          export LD_LIBRARY_PATH="${cudaRoot}/nvvm/lib:${cudaRoot}/nvvm/lib64:${cudaRoot}/lib64:${cudaRoot}/lib:${pkgs.ncurses.out}/lib:${pkgs.libxml2.out}/lib:${pkgs.zlib.out}/lib:${pkgs.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

          echo "rust-cuda llvm19 shell"
          echo "  CUDA_HOME=$CUDA_HOME"
          echo "  LLVM_CONFIG_19=$LLVM_CONFIG_19"
        '';
      };
    in
    {
      devShells.${system}.default = devShell;
    };
}
