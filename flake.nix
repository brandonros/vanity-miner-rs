{
  description = "Stock Rust producer and the llvm-metal compiler";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.rust-overlay.url = "github:oxalica/rust-overlay/1fb104a12a8667045559b2575d6d448ae2fbd99b";
  inputs.rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  # Keep this revision aligned with the llvm-metal crates in the Cargo manifests.
  inputs.llvm-metal.url = "github:brandonros/llvm-metal/1c011e80fae5c9b83fdd44226a69d4173a816c54";

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
              pkgs.python3
            ];
          };
        in { default = shell; metal = shell; });
    };
}
