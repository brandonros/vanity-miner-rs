{ lib, llvmPackages_21, apple-sdk_15, cmake, ninja, pkg-config, lz4, zstd, darwin, source }:

llvmPackages_21.stdenv.mkDerivation {
  pname = "vanity-cumetal";
  version = builtins.substring 0 12 source.rev;
  src = source;

  nativeBuildInputs = [ cmake ninja pkg-config darwin.sigtool ];
  buildInputs = [ apple-sdk_15 llvmPackages_21.llvm.dev lz4 zstd ];
  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CXX_SCAN_FOR_MODULES=OFF"
    "-DCUMETAL_BUILD_TESTS=OFF"
    "-DCUMETAL_ENABLE_BINARY_SHIM=OFF"
    "-DLLVM_DIR=${llvmPackages_21.llvm.dev}/lib/cmake/llvm"
  ];

  postPatch = ''
    # Runtime helpers must refer to immutable sources after the build is removed.
    substituteInPlace CMakeLists.txt \
      --replace-fail 'CUMETAL_SOURCE_DIR="''${CMAKE_SOURCE_DIR}"' 'CUMETAL_SOURCE_DIR="${source}"'
    # Use the signing tool available inside the Nix build sandbox.
    substituteInPlace cmake/prepare_macos_binary.cmake \
      --replace-fail '/usr/bin/codesign' '${darwin.sigtool}/bin/codesign'
  '';

  # The consumer needs only this compiler/runtime pair, built in the same invocation.
  buildPhase = ''
    runHook preBuild
    cmake --build . --target cumetalc cumetal_runtime --parallel "$NIX_BUILD_CORES"
    runHook postBuild
  '';
  installPhase = ''
    runHook preInstall
    mkdir -p "$out/bin" "$out/lib"
    cp cumetalc "$out/bin/"
    cp libcumetal.dylib "$out/lib/"
    runHook postInstall
  '';

  # Hash the final, stripped/signed artifacts, rather than their build-tree copies.
  postFixup = ''
    mkdir -p "$out/share/cumetal"
    compiler_hash=$(sha256sum "$out/bin/cumetalc" | cut -d ' ' -f1)
    runtime_hash=$(sha256sum "$out/lib/libcumetal.dylib" | cut -d ' ' -f1)
    cat > "$out/share/cumetal/build.json" <<EOF
    {"schema":1,"revision":"${source.rev}","source":"https://github.com/brandonros/cuda-metal","source_path":"${source}","compiler_sha256":"$compiler_hash","runtime_sha256":"$runtime_hash"}
    EOF
  '';

  meta = {
    description = "CuMetal compiler and runtime pinned for vanity-miner validation";
    platforms = [ "aarch64-darwin" ];
    license = lib.licenses.asl20;
  };
}
