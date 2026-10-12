{
  lib,
  stdenv,
  cmake,
  ninja,
  autoAddDriverRunpath,
  cudaPackages,
  src,
  llamaSrc,
  version,
  pinChecked ? true,
  cudaArch ? "120",
}:

assert pinChecked;

# strata-vision, the image encoder: tools/vision is its own cmake project (llama.cpp's mtmd plus
# ggml-cuda at the engine's pinned llama.cpp) and `strata-serve` keeps one of these alive beside the
# engine, so the engine binary stays what it is.  Only wanted when a service takes pictures.
cudaPackages.backendStdenv.mkDerivation {
  pname = "strata-vision";
  inherit version;
  inherit src;
  strictDeps = true;

  nativeBuildInputs = [
    cmake
    ninja
    cudaPackages.cuda_nvcc
    autoAddDriverRunpath
  ];

  buildInputs = [ cudaPackages.cudatoolkit ];

  cmakeDir = "../tools/vision";

  cmakeFlags = [
    "-DCMAKE_GENERATOR=Ninja"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DLLAMA_DIR=${llamaSrc}"
    "-DSTRATA_VISION_CUDA=ON"
    # The portable build on purpose (their default): a -march=native encoder built on one PC dies with
    # an illegal instruction on another (#411 #412 #419), and a Nix build runs on a different machine
    # than the one that later runs the result.  The encoder's own CMakeLists forces the AVX2 baseline
    # (and AVX-512 off) with this on, which is what a cached binary wants.
    "-DSTRATA_PORTABLE=ON"
    "-DCMAKE_CUDA_ARCHITECTURES=${cudaArch}"
    "-DCMAKE_CUDA_COMPILER=${cudaPackages.cuda_nvcc}/bin/nvcc"
  ];

  env = {
    CUDAToolkit_ROOT = cudaPackages.cudatoolkit;
    CUDAToolkit_INCLUDE_ROOT = cudaPackages.cudatoolkit;
    CUDAToolkit_LIBRARY_ROOT = cudaPackages.cudatoolkit;
  };

  buildFlags = [ "--target" "strata-vision" ];

  installPhase = ''
    runHook preInstall
    exe=$(find "$NIX_BUILD_TOP" -maxdepth 5 -type f -name strata-vision | head -1)
    if [[ -z "$exe" ]]; then
      echo "strata-vision binary not found under $NIX_BUILD_TOP" >&2
      exit 1
    fi
    install -Dm755 "$exe" $out/bin/strata-vision
    runHook postInstall
  '';

  passthru = {
    inherit cudaArch;
  };

  meta = {
    description = "Strata image encoder (llama.cpp mtmd + mmproj) for Qwen3.8-Flash-Next on NVIDIA (sm_${cudaArch})";
    homepage = "https://github.com/Niko1221/Strata";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "strata-vision";
  };
}
