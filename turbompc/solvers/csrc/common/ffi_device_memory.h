#pragma once

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cuda_runtime.h>
#include <cudss.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace turbompc {

// Valid only for the duration of the current FFI call.
struct FfiDeviceAllocator {
  const XLA_FFI_Api* api;
  XLA_FFI_ExecutionContext* ctx;
};

inline void CheckFfiError(FfiDeviceAllocator& allocator,
                          XLA_FFI_Error* error) {
  if (!error) return;
  XLA_FFI_Error_GetMessage_Args message{};
  message.struct_size = XLA_FFI_Error_GetMessage_Args_STRUCT_SIZE;
  message.error = error;
  allocator.api->XLA_FFI_Error_GetMessage(&message);
  std::string text = message.message;
  XLA_FFI_Error_Destroy_Args destroy{};
  destroy.struct_size = XLA_FFI_Error_Destroy_Args_STRUCT_SIZE;
  destroy.error = error;
  allocator.api->XLA_FFI_Error_Destroy(&destroy);
  throw std::runtime_error("XLA device memory: " + text);
}

inline size_t AllocationSize(size_t bytes) { return bytes == 0 ? 1 : bytes; }

template <typename T>
inline void FfiAllocate(FfiDeviceAllocator& allocator, T** pointer, size_t bytes) {
  XLA_FFI_DeviceMemory_Allocate_Args args{};
  args.struct_size = XLA_FFI_DeviceMemory_Allocate_Args_STRUCT_SIZE;
  args.ctx = allocator.ctx;
  args.size = AllocationSize(bytes);
  args.alignment = 16;
  CheckFfiError(allocator, allocator.api->XLA_FFI_DeviceMemory_Allocate(&args));
  *pointer = static_cast<T*>(args.data);
}

inline void FfiFree(FfiDeviceAllocator& allocator, void* pointer, size_t bytes) {
  if (!pointer) return;
  XLA_FFI_DeviceMemory_Free_Args args{};
  args.struct_size = XLA_FFI_DeviceMemory_Free_Args_STRUCT_SIZE;
  args.ctx = allocator.ctx;
  args.size = AllocationSize(bytes);
  args.data = pointer;
  CheckFfiError(allocator, allocator.api->XLA_FFI_DeviceMemory_Free(&args));
}

inline void ReportAllocatorError(const char* text) {
  std::fprintf(stderr, "TurboMPC device memory: %s\n", text);
}

// cuDSS retains the handler context, so it must point to storage owned by the
// workspace. The workspace refreshes its two values before each cuDSS call.
inline cudssStatus_t SetCudssDeviceMemory(cudssHandle_t handle, FfiDeviceAllocator& allocator) {
  cudssDeviceMemHandler_t handler{};
  handler.ctx = &allocator;
  std::strcpy(handler.name, "TurboMPC XLA FFI allocator");
  handler.device_alloc = [](void* context, void** pointer, size_t size,
                            cudaStream_t) -> int {
    try {
      if (size == 0) {
        *pointer = nullptr;
        return 0;
      }
      FfiAllocate(*static_cast<FfiDeviceAllocator*>(context), pointer, size);
      return 0;
    } catch (const std::exception& error) {
      ReportAllocatorError(error.what());
      return 1;
    }
  };
  handler.device_free = [](void* context, void* pointer, size_t size,
                           cudaStream_t stream) -> int {
    if (!pointer) return 0;
    const cudaError_t status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
      ReportAllocatorError(cudaGetErrorString(status));
      return 1;
    }
    try {
      FfiFree(*static_cast<FfiDeviceAllocator*>(context), pointer, size);
      return 0;
    } catch (const std::exception& error) {
      ReportAllocatorError(error.what());
      return 1;
    }
  };
  return cudssSetDeviceMemHandler(handle, &handler);
}

}  // namespace turbompc

namespace xla::ffi {
template <>
struct CtxDecoding<turbompc::FfiDeviceAllocator> {
  using Type = turbompc::FfiDeviceAllocator;
  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine&) {
    return Type{api, ctx};
  }
};
}  // namespace xla::ffi
