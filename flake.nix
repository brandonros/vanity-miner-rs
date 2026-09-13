{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    # LLVM 7 is no longer carried by nixpkgs-unstable. Pin a second nixpkgs just
    # for `llvmPackages_7` so someone else's compat patches do the hard work.
    nixpkgs-llvm7.url = "github:NixOS/nixpkgs/nixos-23.05";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { nixpkgs, nixpkgs-llvm7, rust-overlay, ... }:
    let
      systems = [ "aarch64-linux" "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;

      mkDevShell = system: version:
        let
          # allowUnfree is required because CUDA is unfree.
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
            overlays = [ rust-overlay.overlays.default ];
          };
          # Old nixpkgs solely so we can fish out llvmPackages_7 *and* the libs
          # it links against. Critical: anything that lands on LD_LIBRARY_PATH
          # while clang 7 is running must come from this set — clang 7's glibc
          # is 2.37, and unstable's libstdc++ / ncurses now demand 2.38+/2.42.
          pkgsLlvm7 = import nixpkgs-llvm7 { inherit system; };
          lib = pkgs.lib;
          # Match the C toolchain and its runtime libraries to the selected LLVM.
          compatPkgs = if version == 7 then pkgsLlvm7 else pkgs;
          llvm = if version == 7 then pkgsLlvm7.llvmPackages_7 else pkgs.llvmPackages_19;

          # ---- CUDA toolkit (Nix-managed) ----
          # CUDA 13.2 → NVVM 22.0 → PTX 9.2 → needs driver 580.x+ (CUDA 13) at runtime.
          # `cudatoolkit` is the kitchen-sink symlinkJoin maintained by nixpkgs —
          # every header path and lib layout is already wired correctly. The host
          # NVIDIA driver (libcuda.so.1) is needed at runtime; it is *not* shimmed
          # in here — supply it via the system or extend LD_LIBRARY_PATH yourself
          # before running CUDA programs.
          cudaRoot = pkgs.cudaPackages_13_2.cudatoolkit;

          # Single source of truth for channel + components lives in
          # rust-toolchain.toml. Update there, not here.
          toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

          # Versioned tools used by the Rust-CUDA backend.
          llvmBin = lib.getBin llvm.llvm;
          llvmDev = lib.getDev llvm.llvm;
          llvmCompatTools = pkgs.symlinkJoin {
            name = "llvm${toString version}-compat-tools";
            paths = [
              (pkgs.writeShellScriptBin "opt-${toString version}" ''exec ${llvmBin}/bin/opt "$@"'')
              (pkgs.writeShellScriptBin "llvm-as-${toString version}" ''exec ${llvmBin}/bin/llvm-as "$@"'')
              (pkgs.writeShellScriptBin "llvm-dis-${toString version}" ''exec ${llvmBin}/bin/llvm-dis "$@"'')
              (pkgs.writeShellScriptBin "llc-${toString version}" ''exec ${llvmBin}/bin/llc "$@"'')
            ];
          };
        in
        pkgs.mkShell {
          CUDA_HOME = "${cudaRoot}";
          CUDA_ROOT = "${cudaRoot}";
          CUDA_PATH = "${cudaRoot}";
          CUDA_TOOLKIT_ROOT_DIR = "${cudaRoot}";
          # Cover both lib/ (nix-style) and lib64/ (FHS-style) so downstream
          # build.rs scripts that probe either layout resolve libcudart + stubs.
          CUDA_LIBRARY_PATH =
            "${cudaRoot}/lib:${cudaRoot}/lib64:${cudaRoot}/lib/stubs:${cudaRoot}/lib64/stubs";
          ${if version == 7 then "LLVM_CONFIG" else "LLVM_CONFIG_19"} = "${llvmDev}/bin/llvm-config";
          LIBCLANG_PATH = "${lib.getLib llvm.libclang}/lib";

          # nativeBuildInputs: tools invoked *during* the build — compilers,
          # codegen, build systems. End up on $PATH. cudaRoot lives here because
          # build-cuda.sh shells out to ptxas/nvcc; its libraries are picked up
          # via LD_LIBRARY_PATH/LIBRARY_PATH below.
          # No pkgs.gcc — clang 7 is our C compiler. Pulling gcc-15 from
          # unstable would put its libstdc++ on the link/runtime path and
          # reintroduce the GLIBC_2.42 mismatch.
          nativeBuildInputs = [
            toolchain
            pkgs.pkg-config
            pkgs.cmake
            pkgs.ninja
            pkgs.patchelf
            cudaRoot
            llvm.clang
            llvm.libclang
            llvmBin
            llvmDev
            llvmCompatTools
          ];
          # LLVM 7 uses the old package set; LLVM 19 uses current libraries.
          buildInputs = [
            compatPkgs.openssl
            compatPkgs.libxml2
            compatPkgs.zlib
            compatPkgs.ncurses
            compatPkgs.libffi
            compatPkgs.stdenv.cc.cc.lib
          ];

          shellHook = ''
            export CARGO_TARGET_DIR="$PWD/target/llvm${toString version}"
            export PATH="${llvmCompatTools}/bin:${llvmBin}/bin:${llvmDev}/bin:${cudaRoot}/bin:${cudaRoot}/nvvm/bin:$PATH"
            export LD_LIBRARY_PATH="${cudaRoot}/nvvm/lib:${cudaRoot}/nvvm/lib64:${cudaRoot}/lib64:${cudaRoot}/lib:${compatPkgs.ncurses.out}/lib:${compatPkgs.libxml2.out}/lib:${compatPkgs.zlib.out}/lib:${compatPkgs.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            # LIBRARY_PATH is the *link-time* analog of LD_LIBRARY_PATH — needed
            # so cc/ld can resolve `-lnvvm` (from #[link(name = "nvvm")]) etc.
            export LIBRARY_PATH="${cudaRoot}/nvvm/lib64:${cudaRoot}/nvvm/lib:${cudaRoot}/lib64:${cudaRoot}/lib''${LIBRARY_PATH:+:$LIBRARY_PATH}"

            echo "rust-cuda llvm${toString version} shell"
            echo "  CUDA_HOME=$CUDA_HOME"
            echo "  LLVM_CONFIG=${llvmDev}/bin/llvm-config"
          '';
        };
    in
    {
      devShells = forAllSystems (system: {
        default = mkDevShell system 7;
        v7 = mkDevShell system 7;
        v19 = mkDevShell system 19;
      });
    };
}
