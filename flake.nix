{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { nixpkgs, rust-overlay, ... }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
    in
    {
      devShells = nixpkgs.lib.genAttrs systems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
            config.allowUnfree = true; # CUDA
          };
          rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
          # cust generates driver API bindings from these headers.
          cuda = pkgs.cudaPackages_12_9.cudatoolkit;
        in
        {
          default = pkgs.mkShell ({
            packages = [ rust ];
          } // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
            # bindgenHook points cust's bindgen at libclang and the C headers.
            nativeBuildInputs = [ pkgs.patchelf pkgs.rustPlatform.bindgenHook ];
            CUDA_PATH = "${cuda}";
          });
        });
    };
}
