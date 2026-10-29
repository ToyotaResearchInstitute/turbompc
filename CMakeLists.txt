cmake_minimum_required(VERSION 3.20)

# Entry point for:
#   - the Eigen pybind11 extension (CPU)
#   - the CUDA FFI libraries (turbompc/solvers/csrc)
#
# pip uses this file through scikit-build-core.

project(turbompc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ── Component options ─────────────────────────────────────────────
# AUTO builds the component when its toolchain is present,
# ON requires toolchain and hard error when missing
# OFF skips it.
# turbompc_build.py also reads TURBOMPC_CUDA to select build dependencies.
set(TURBOMPC_CUDA "AUTO" CACHE STRING "CUDA FFI libraries: AUTO, ON, or OFF")
set(TURBOMPC_EIGEN "AUTO" CACHE STRING "Eigen extension: AUTO, ON, or OFF")

macro(turbompc_resolve_tristate name)
  if(DEFINED ENV{${name}} AND NOT "$ENV{${name}}" STREQUAL "")
    set(${name} "$ENV{${name}}")
  endif()
  string(STRIP "${${name}}" ${name})
  string(TOUPPER "${${name}}" ${name})
  if(${name} MATCHES "^(1|TRUE|YES)$")
    set(${name} "ON")
  elseif(${name} MATCHES "^(0|FALSE|NO)$")
    set(${name} "OFF")
  endif()
  if(NOT ${name} MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "${name} must be AUTO, ON, or OFF (got '${${name}}')")
  endif()
endmacro()

turbompc_resolve_tristate(TURBOMPC_CUDA)
turbompc_resolve_tristate(TURBOMPC_EIGEN)

# ── Python and the XLA FFI headers ────────────────────────────────
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Both the Eigen extension and the CUDA libraries compile against jaxlib's
# header-only XLA FFI API, so this is required even for a CPU-only build.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "import os, jaxlib; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))"
  OUTPUT_VARIABLE JAXLIB_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _jaxlib_rc
  ERROR_QUIET
)
if(NOT _jaxlib_rc EQUAL 0 OR NOT JAXLIB_INCLUDE_DIR)
  message(FATAL_ERROR
    "Could not import jaxlib to locate the XLA FFI headers, using\n"
    "  ${Python3_EXECUTABLE}\n"
    "jaxlib is a build dependency (see [build-system].requires in pyproject.toml). "
    "If you are configuring cmake directly, or building with --no-build-isolation, "
    "install it first:  pip install jaxlib")
endif()
message(STATUS "XLA FFI include: ${JAXLIB_INCLUDE_DIR}")

# ── CUDA ──────────────────────────────────────────────────────────
# check_language() checks for a compiler, so a missing toolkit means
# does not fail the build (assumes CPU).
include(CheckLanguage)
set(TURBOMPC_CUDA_ENABLED OFF)
if(TURBOMPC_CUDA STREQUAL "OFF")
  message(STATUS "CUDA: disabled (TURBOMPC_CUDA=OFF)")
else()
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(TURBOMPC_CUDA_ENABLED ON)
  elseif(TURBOMPC_CUDA STREQUAL "ON")
    message(FATAL_ERROR
      "TURBOMPC_CUDA=ON but no CUDA compiler was found.\n"
      "  Install the CUDA Toolkit (nvcc >= 12), or\n"
      "  configure with -DTURBOMPC_CUDA=OFF to build the CPU backends only.")
  else()
    message(STATUS "CUDA: no toolkit found: building CPU backends only")
  endif()
endif()

if(TURBOMPC_CUDA_ENABLED)
  # GPU architecture detection must happen BEFORE enable_language(CUDA).
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
        # "8.9" -> "89", "12.0" -> "120"
        string(REGEX MATCH "([0-9]+)\\.([0-9]+)" _MATCH "${_GPU_CC}")
        set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
      else()
        # Fallback: build for all common GPUs (Turing through Blackwell)
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90;120")
      endif()
    endif()
  endif()
  # Persist the resolved value so users can verify it:
  #   grep CMAKE_CUDA_ARCHITECTURES build/skbuild/*/CMakeCache.txt
  set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      CACHE STRING "CUDA architectures" FORCE)

  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

  find_package(CUDAToolkit REQUIRED)

  add_subdirectory(turbompc/solvers/csrc)
endif()

# ── Eigen extension (CPU block-tridiagonal Schur solve) ───────────
set(TURBOMPC_EIGEN_INSTALL_DIR "turbompc/solvers/linear_systems_solvers")

set(TURBOMPC_EIGEN_ENABLED OFF)
if(TURBOMPC_EIGEN STREQUAL "OFF")
  message(STATUS "Eigen: disabled (TURBOMPC_EIGEN=OFF)")
else()
  # Prefer EIGEN_INCLUDE_DIR from the CMake cache or environment, then a
  # packaged Eigen3 config, then the Debian/Ubuntu default.
  if(NOT EIGEN_INCLUDE_DIR AND DEFINED ENV{EIGEN_INCLUDE_DIR})
    set(EIGEN_INCLUDE_DIR "$ENV{EIGEN_INCLUDE_DIR}")
  endif()
  if(NOT EIGEN_INCLUDE_DIR)
    find_package(Eigen3 3.3 QUIET NO_MODULE)
    if(Eigen3_FOUND)
      get_target_property(EIGEN_INCLUDE_DIR Eigen3::Eigen
                          INTERFACE_INCLUDE_DIRECTORIES)
    else()
      set(EIGEN_INCLUDE_DIR "/usr/include/eigen3")
    endif()
  endif()

  # pybind11 is part of the Eigen toolchain, so a missing pybind11 is treated
  # exactly like missing Eigen headers: skipped under AUTO, fatal under ON.
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_rc
    ERROR_QUIET
  )
  if(_pybind11_rc EQUAL 0 AND _pybind11_cmakedir)
    list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmakedir}")
    find_package(pybind11 CONFIG QUIET)
  endif()

  if(NOT EXISTS "${EIGEN_INCLUDE_DIR}/Eigen/Core")
    set(_eigen_missing "Eigen headers not found at ${EIGEN_INCLUDE_DIR}")
    set(_eigen_fix
      "  Install them:  sudo apt install libeigen3-dev\n"
      "  Or point at them:  -DEIGEN_INCLUDE_DIR=<path>")
  elseif(NOT pybind11_FOUND)
    set(_eigen_missing "pybind11 not found via ${Python3_EXECUTABLE}")
    set(_eigen_fix "  Install it:  pip install pybind11")
  else()
    set(TURBOMPC_EIGEN_ENABLED ON)
  endif()

  if(NOT TURBOMPC_EIGEN_ENABLED)
    if(TURBOMPC_EIGEN STREQUAL "ON")
      message(FATAL_ERROR
        "TURBOMPC_EIGEN=ON but ${_eigen_missing}.\n"
        ${_eigen_fix} "\n"
        "  Or skip the extension:  -DTURBOMPC_EIGEN=OFF")
    endif()
    message(STATUS
      "Eigen: ${_eigen_missing} - skipping the Eigen extension "
      "(ADMM_JAX_LOOP_EIGEN will be unavailable)")
  endif()
endif()

if(TURBOMPC_EIGEN_ENABLED)
  message(STATUS "Eigen include: ${EIGEN_INCLUDE_DIR}")
  pybind11_add_module(_eigen_block_tridi
    turbompc/solvers/linear_systems_solvers/eigen_block_tridi.cpp
  )
  target_include_directories(_eigen_block_tridi SYSTEM PRIVATE
    ${JAXLIB_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
  )
  install(TARGETS _eigen_block_tridi
    LIBRARY DESTINATION ${TURBOMPC_EIGEN_INSTALL_DIR}
  )
endif()

# ── Configuration summary ─────────────────────────────────────────
message(STATUS "")
message(STATUS "turbompc configuration:")
message(STATUS "  CUDA FFI libraries : ${TURBOMPC_CUDA_ENABLED} (TURBOMPC_CUDA=${TURBOMPC_CUDA})")
message(STATUS "  Eigen extension    : ${TURBOMPC_EIGEN_ENABLED} (TURBOMPC_EIGEN=${TURBOMPC_EIGEN})")
message(STATUS "Verify the installed result with: turbompc-check")
message(STATUS "")
