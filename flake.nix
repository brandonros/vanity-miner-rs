{
  description = "Stock Rust and LLVM-to-Metal development environment";
  inputs.llvm-metal.url = "github:brandonros/llvm-metal/2d7e7304091ccf6b309cf4c48bea384115c3dbc0";

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
