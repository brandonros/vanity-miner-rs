{
  description = "Stock Rust and LLVM-to-Metal development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.rust-overlay.url = "github:oxalica/rust-overlay/1fb104a12a8667045559b2575d6d448ae2fbd99b";
  inputs.rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  inputs.llvm-downgrade = {
    url = "github:JuliaLLVM/llvm-downgrade/09c2e50d4526b08f55010f6c091ace35332a78a8";
    flake = false;
  };
  # Compiler source only; scripts/build-metal.py builds llvm-metalc from it.
  inputs.llvm-metal = {
    url = "github:brandonros/llvm-metal/df7f9bc4369cfddbdc5ac235154e24e60c8ba4d7";
    flake = false;
  };

  outputs = { nixpkgs, rust-overlay, llvm-downgrade, llvm-metal, ... }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forEachSystem = nixpkgs.lib.genAttrs systems;
    in {
      devShells = forEachSystem (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          # Must be the LLVM major llvm-metal links and rust-toolchain.toml ships.
          llvm = pkgs.llvmPackages_22.llvm;
          downgrade = pkgs.stdenv.mkDerivation {
            pname = "llvm-downgrade";
            version = "09c2e50";
            src = llvm-downgrade;
            nativeBuildInputs = [ pkgs.cmake pkgs.ninja ];
            buildInputs = [ llvm pkgs.libffi ];
            cmakeFlags = [
              "-DLLVM_DIR=${llvm.dev}/lib/cmake/llvm"
              "-DLLVMDG_LINK_DYLIB=ON"
              "-DLLVMDG_BUILD_TESTS=OFF"
            ];
          };
          toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
          shell = pkgs.mkShell {
            packages = [ toolchain llvm downgrade pkgs.python3 ];
            buildInputs = [ pkgs.libffi ];
            LLVM_SYS_221_PREFIX = "${llvm.dev}";
            VANITY_LLVM_METAL_SOURCE = "${llvm-metal}";
          };
        in { default = shell; metal = shell; });
    };
}
