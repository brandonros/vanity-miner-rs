{
  description = "Stock Rust producer and the llvm-metal compiler";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.rust-overlay.url = "github:oxalica/rust-overlay/1fb104a12a8667045559b2575d6d448ae2fbd99b";
  inputs.rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  # Keep this revision aligned with the llvm-metal crates in the Cargo manifests.
  inputs.llvm-metal.url = "github:brandonros/llvm-metal/4b009b23236f04dce90ad9fec795e03c05bd442f";

  outputs = { nixpkgs, rust-overlay, llvm-metal, ... }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
    in {
      devShells = nixpkgs.lib.genAttrs systems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          shell = pkgs.mkShell {
            packages = [
              (pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml)
              # The compiler with its own LLVM and llvm-downgrade.
              llvm-metal.packages.${system}.llvm-metalc
              pkgs.just
            ];
          };
        in { default = shell; metal = shell; });
    };
}
