{
  description = "Stock Rust and LLVM-to-Metal development environment";
  inputs.llvm-metal.url = "github:brandonros/llvm-metal/154ec386346f5dee9b9ffb290d82eea0837ca4f9";

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
