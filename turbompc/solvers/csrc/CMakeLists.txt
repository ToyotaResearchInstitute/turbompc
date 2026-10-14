# CUDA FFI libraries. This file is a subdirectory of the top-level
# CMakeLists.txt. It is only added when CUDA is enabled.

# ── Build options ─────────────────────────────────────────────────
set(TURBOMPC_FFI_INSTALL_DIR "turbompc/solvers/_ffi")
option(TURBOMPC_CUDSS "Build the cuDSS-backed FFI libraries" ON)
option(TURBOMPC_POST_BUILD_CHECK "Run the GPU smoke test after building" ON)

# ── Source directory shortcuts ────────────────────────────────────
# CSRC files are collocated with their Python packages:
set(ADMM_CSRC        ${CMAKE_CURRENT_SOURCE_DIR}/../admm/csrc)
set(LINSYS_CSRC      ${CMAKE_CURRENT_SOURCE_DIR}/../linear_systems_solvers/csrc)
set(BACKWARD_CSRC    ${CMAKE_CURRENT_SOURCE_DIR}/../backward/csrc)
set(COMMON_CSRC      ${CMAKE_CURRENT_SOURCE_DIR}/common)

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

# ---------- cuDSS FFI shared libs ----------
# Auto-detect cuDSS from pip nvidia-cudss package
if(TURBOMPC_CUDSS)
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
  # Cached find results can point into pip's deleted /tmp/pip-build-env-*;
  # drop stale paths so they are re-searched.
  if(CUDSS_LIB AND NOT EXISTS "${CUDSS_LIB}")
    unset(CUDSS_LIB CACHE)
  endif()
  if(CUDSS_INCLUDE_DIR AND NOT EXISTS "${CUDSS_INCLUDE_DIR}/cudss.h")
    unset(CUDSS_INCLUDE_DIR CACHE)
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
endif()
if(TURBOMPC_CUDSS AND CUDSS_LIB)
  message(STATUS "cuDSS found: ${CUDSS_LIB}")
  get_filename_component(_CUDSS_LIB_DIR "${CUDSS_LIB}" DIRECTORY)
  get_filename_component(_NVIDIA_ROOT "${NVIDIA_CU12_DIR}" DIRECTORY)
  file(GLOB _NVIDIA_WHEEL_LIB_DIRS "${_NVIDIA_ROOT}/*/lib")
  list(APPEND _NVIDIA_WHEEL_LIB_DIRS "${_CUDSS_LIB_DIR}")
  list(REMOVE_DUPLICATES _NVIDIA_WHEEL_LIB_DIRS)
  foreach(_dir IN LISTS _NVIDIA_WHEEL_LIB_DIRS)
    add_link_options("-Wl,-rpath-link,${_dir}")
  endforeach()
  find_path(CUDSS_INCLUDE_DIR cudss.h HINTS
    ${CUDSS_DIR}/include
    $ENV{CUDSS_DIR}/include
    ${NVIDIA_CU12_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    /usr/local/cuda/include
  )
  # Compile the reusable solver into each FFI library. This keeps either FFI
  # library independently loadable and avoids an extra runtime .so dependency.
  add_library(cudss_blktridi_core STATIC
    ${LINSYS_CSRC}/cudss_blktridi.cu
  )
  target_include_directories(cudss_blktridi_core PRIVATE
    ${JAXLIB_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDSS_INCLUDE_DIR}
    ${COMMON_CSRC}
    ${LINSYS_CSRC}
  )
  target_link_libraries(cudss_blktridi_core PRIVATE
    blktridi_common CUDA::cudart CUDA::cusparse ${CUDSS_LIB}
  )
  set_target_properties(cudss_blktridi_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )

  add_library(cudss_blktridi_ffi SHARED
    ${LINSYS_CSRC}/cudss_ffi.cc
    ${LINSYS_CSRC}/cudss_workspace_ffi.cc
  )
  target_include_directories(cudss_blktridi_ffi SYSTEM PRIVATE
    ${JAXLIB_INCLUDE_DIR}
  )
  target_include_directories(cudss_blktridi_ffi PRIVATE
    ${COMMON_CSRC}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDSS_INCLUDE_DIR}
    ${LINSYS_CSRC}
  )
  target_link_libraries(cudss_blktridi_ffi PRIVATE cudss_blktridi_core)
  set_target_properties(cudss_blktridi_ffi PROPERTIES
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
elseif(TURBOMPC_CUDSS)
  message(FATAL_ERROR
    "cuDSS not found. ADMM_FUSED_CUDSS and DIRECT_CUDSS_FFI are the TurboMPCSolver "
    "defaults, so building without them produces an install whose default "
    "configuration fails at runtime.\n"
    "  Fix:  pip install 'nvidia-cudss-cu12>=0.8'\n"
    "  Or:   configure with -DCUDSS_DIR=<path> (https://developer.nvidia.com/cudss)\n"
    "  Or:   configure with -DTURBOMPC_CUDSS=OFF for a deliberate PCG-only build")
else()
  message(STATUS "TURBOMPC_CUDSS=OFF — skipping cuDSS FFI targets")
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
# Built via a direct cmake configure of the repo root (see docs/cuda_backends.md):
#   cmake --build <build-dir> --target admm_fused_profile_ffi
# Use: ADMM_FFI_LIB=<build-dir>/libadmm_fused_profile_ffi.so python ...
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
if(TURBOMPC_POST_BUILD_CHECK)
  add_custom_command(
    TARGET admm_fused_ffi POST_BUILD
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/post_build_check.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running post-build smoke test..."
    VERBATIM
  )
endif()

# ---------- Install rules ----------
# Shared FFI libraries install into the Python package so that pip-installed
# copies are self-contained.
set(TURBOMPC_FFI_TARGETS
  pcg_blktridi_ffi
  admm_fused_ffi
  admm_fused_profile_ffi
)
if(TURBOMPC_CUDSS AND CUDSS_LIB)
  list(APPEND TURBOMPC_FFI_TARGETS
    cudss_blktridi_ffi
    admm_cudss_ffi
    cudss_sparse_kkt_ffi
  )
endif()
install(TARGETS ${TURBOMPC_FFI_TARGETS}
  LIBRARY DESTINATION ${TURBOMPC_FFI_INSTALL_DIR}
)
