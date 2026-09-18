{
  description = "Stock Rust and LLVM-to-Metal development environment";
  inputs.llvm-metal.url = "github:brandonros/llvm-metal/9cbb88e2b54e4719fb833eca14fb5f41421ebad0";

  outputs = { llvm-metal, ... }: {
    devShells = builtins.mapAttrs (system: shells:
      let
        pkgs = import llvm-metal.inputs.nixpkgs {
          inherit system;
          overlays = [ llvm-metal.inputs.rust-overlay.overlays.default ];
        };
        toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        shell = shells.rust-fixtures.overrideAttrs (old: {
          nativeBuildInputs = [ toolchain ] ++ old.nativeBuildInputs;
          VANITY_LLVM_METAL_SOURCE = "${llvm-metal}";
        });
      in { default = shell; metal = shell; }
    ) llvm-metal.devShells;
  };
}
