cmake_minimum_required(VERSION 3.20)

# ── GPU architecture detection ───────────────────────────────────
# Must happen BEFORE project() — cmake sets CMAKE_CUDA_ARCHITECTURES
# to a useless default during enable_language(CUDA) otherwise.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    # cmake < 3.24: auto-detect via nvidia-smi
    execute_process(
      COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE _GPU_CC OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _GPU_CC_RESULT
    )
    if(_GPU_CC_RESULT EQUAL 0 AND _GPU_CC)
      # "8.9" → "89", "12.0" → "120"
      string(REGEX MATCH "([0-9]+)\\.([0-9]+)" _MATCH "${_GPU_CC}")
      set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
    else()
      # Fallback: build for all common GPUs (Turing through Blackwell)
      set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90;120")
    endif()
  endif()
endif()

project(turbompc_ffi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
# Persist resolved value so users can verify: grep CMAKE_CUDA_ARCHITECTURES build/ffi/CMakeCache.txt
set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "import os, jaxlib; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))"
  OUTPUT_VARIABLE JAXLIB_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "XLA FFI include: ${JAXLIB_INCLUDE_DIR}")

find_package(CUDAToolkit REQUIRED)

# ── Source directory shortcuts ────────────────────────────────────
# CSRC files are collocated with their Python packages:
#   admm kernels:           ../admm/csrc/
#   linear solver kernels:  ../linear_systems_solvers/csrc/
#   backward KKT kernels:   ../backward/csrc/
#   shared utilities:       common/
set(ADMM_CSRC      ${CMAKE_CURRENT_SOURCE_DIR}/../admm/csrc)
set(LINSYS_CSRC    ${CMAKE_CURRENT_SOURCE_DIR}/../linear_systems_solvers/csrc)
set(BACKWARD_CSRC  ${CMAKE_CURRENT_SOURCE_DIR}/../backward/csrc)
set(COMMON_CSRC    ${CMAKE_CURRENT_SOURCE_DIR}/common)

# ---------- common static lib ----------
add_library(blktridi_common STATIC
  ${COMMON_CSRC}/block_tridi.cu
)
target_include_directories(blktridi_common PUBLIC
  ${COMMON_CSRC}
  ${CUDAToolkit_INCLUDE_DIRS}
)
set_target_properties(blktridi_common PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# ---------- PCG FFI shared lib ----------
add_library(pcg_blktridi_ffi SHARED
  ${LINSYS_CSRC}/pcg_blktridi.cu
  ${LINSYS_CSRC}/pcg_ffi.cc
)
target_include_directories(pcg_blktridi_ffi SYSTEM PRIVATE
  ${JAXLIB_INCLUDE_DIR}
)
target_include_directories(pcg_blktridi_ffi PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
  ${COMMON_CSRC}
  ${LINSYS_CSRC}
)
target_link_libraries(pcg_blktridi_ffi PRIVATE blktridi_common CUDA::cudart)
set_target_properties(pcg_blktridi_ffi PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# ---------- cuDSS FFI shared lib (optional) ----------
# Auto-detect cuDSS from pip nvidia-cudss package
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "import nvidia.cu12; print(list(nvidia.cu12.__path__)[0])"
  OUTPUT_VARIABLE NVIDIA_CU12_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE _nvidia_cu12_rc
)
if(_nvidia_cu12_rc EQUAL 0)
  message(STATUS "nvidia.cu12 package: ${NVIDIA_CU12_DIR}")
endif()
# Prefer pip nvidia-cudss-cu12 (libcublas-compatible) over system .deb cuDSS.
find_library(CUDSS_LIB cudss HINTS
  ${CUDSS_DIR}/lib
  $ENV{CUDSS_DIR}/lib
  ${NVIDIA_CU12_DIR}/lib
  NO_DEFAULT_PATH
)
if(NOT CUDSS_LIB)
  # pip nvidia-cudss-cu12 ships libcudss.so.0 without an unversioned symlink
  file(GLOB _CUDSS_SO0
    "${NVIDIA_CU12_DIR}/lib/libcudss.so.*"
    "${CUDSS_DIR}/lib/libcudss.so.*"
    "$ENV{CUDSS_DIR}/lib/libcudss.so.*"
  )
  if(_CUDSS_SO0)
    list(GET _CUDSS_SO0 0 CUDSS_LIB)
    message(STATUS "cuDSS found (versioned, pip): ${CUDSS_LIB}")
  endif()
endif()
if(NOT CUDSS_LIB)
  # Last resort: system paths (may have incompatible libcublas version)
  find_library(CUDSS_LIB cudss HINTS
    ${CUDAToolkit_LIBRARY_DIR}
    /usr/local/cuda/lib64
  )
endif()
if(CUDSS_LIB)
  message(STATUS "cuDSS found: ${CUDSS_LIB}")
  find_path(CUDSS_INCLUDE_DIR cudss.h HINTS
    ${CUDSS_DIR}/include
    $ENV{CUDSS_DIR}/include
    ${NVIDIA_CU12_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    /usr/local/cuda/include
  )
  add_library(cudss_blktridi_ffi SHARED
    ${LINSYS_CSRC}/cudss_blktridi.cu
    ${LINSYS_CSRC}/cudss_ffi.cc
  )
  target_include_directories(cudss_blktridi_ffi SYSTEM PRIVATE
    ${JAXLIB_INCLUDE_DIR}
  )
  target_include_directories(cudss_blktridi_ffi PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDSS_INCLUDE_DIR}
    ${COMMON_CSRC}
    ${LINSYS_CSRC}
  )
  target_link_libraries(cudss_blktridi_ffi PRIVATE
    blktridi_common CUDA::cudart CUDA::cusparse ${CUDSS_LIB}
  )
  set_target_properties(cudss_blktridi_ffi PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )

  # ---------- ADMM-cuDSS FFI shared lib ----------
  add_library(admm_cudss_ffi SHARED
    ${ADMM_CSRC}/admm_cudss.cu
    ${ADMM_CSRC}/admm_cudss_ffi.cc
  )
  target_include_directories(admm_cudss_ffi SYSTEM PRIVATE
    ${JAXLIB_INCLUDE_DIR}
  )
  target_include_directories(admm_cudss_ffi PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDSS_INCLUDE_DIR}
    ${COMMON_CSRC}
    ${ADMM_CSRC}
  )
  target_link_libraries(admm_cudss_ffi PRIVATE
    blktridi_common CUDA::cudart CUDA::cusparse ${CUDSS_LIB}
  )
  set_target_properties(admm_cudss_ffi PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )

  # ---------- cuDSS Sparse KKT FFI shared lib (for backward pass) ----------
  add_library(cudss_sparse_kkt_ffi SHARED
    ${BACKWARD_CSRC}/cudss_sparse_kkt.cu
    ${BACKWARD_CSRC}/cudss_sparse_kkt_ffi.cc
  )
  target_include_directories(cudss_sparse_kkt_ffi SYSTEM PRIVATE
    ${JAXLIB_INCLUDE_DIR}
  )
  target_include_directories(cudss_sparse_kkt_ffi PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDSS_INCLUDE_DIR}
    ${COMMON_CSRC}
    ${BACKWARD_CSRC}
  )
  target_link_libraries(cudss_sparse_kkt_ffi PRIVATE
    CUDA::cudart CUDA::cusparse ${CUDSS_LIB}
  )
  set_target_properties(cudss_sparse_kkt_ffi PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )
else()
  message(WARNING "cuDSS not found — skipping cudss_blktridi_ffi, admm_cudss_ffi, and cudss_sparse_kkt_ffi targets. "
    "Install cuDSS (https://developer.nvidia.com/cudss) and set -DCUDSS_DIR=<path> to enable.")
endif()

# ---------- ADMM Fused FFI shared lib ----------
add_library(admm_fused_ffi SHARED
  ${ADMM_CSRC}/admm_fused.cu
  ${ADMM_CSRC}/admm_ffi.cc
)
target_include_directories(admm_fused_ffi SYSTEM PRIVATE
  ${JAXLIB_INCLUDE_DIR}
)
target_include_directories(admm_fused_ffi PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
  ${COMMON_CSRC}
  ${LINSYS_CSRC}
  ${ADMM_CSRC}
)
target_link_libraries(admm_fused_ffi PRIVATE blktridi_common CUDA::cudart)
set_target_properties(admm_fused_ffi PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# ---------- ADMM Fused FFI (profiling variant) ----------
# Build with: cmake --build build/ffi --target admm_fused_profile_ffi
# Use: ADMM_FFI_LIB=build/ffi/libadmm_fused_profile_ffi.so python ...
add_library(admm_fused_profile_ffi SHARED
  ${ADMM_CSRC}/admm_fused.cu
  ${ADMM_CSRC}/admm_ffi.cc
)
target_include_directories(admm_fused_profile_ffi SYSTEM PRIVATE
  ${JAXLIB_INCLUDE_DIR}
)
target_include_directories(admm_fused_profile_ffi PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
  ${COMMON_CSRC}
  ${LINSYS_CSRC}
  ${ADMM_CSRC}
)
target_link_libraries(admm_fused_profile_ffi PRIVATE blktridi_common CUDA::cudart)
target_compile_definitions(admm_fused_profile_ffi PRIVATE ADMM_PROFILE)
set_target_properties(admm_fused_profile_ffi PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# ---------- Post-build smoke test ----------
# Runs automatically after build to catch NaN, stale libs, architecture mismatches.
# Uses the Python interpreter that JAX is installed in.
add_custom_command(
  TARGET admm_fused_ffi POST_BUILD
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/post_build_check.py
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Running post-build smoke test..."
  VERBATIM
)
