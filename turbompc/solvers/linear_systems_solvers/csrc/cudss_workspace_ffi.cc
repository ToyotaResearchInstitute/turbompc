#include "ffi_device_memory.h"

#include <cstdint>

namespace ffi = xla::ffi;

std::uint64_t CreateCudssBlockTridiWorkspace();
int EnsureCudssBlockTridiWorkspaceCapacity(turbompc::FfiDeviceAllocator&, std::uint64_t, int32_t, int32_t, int32_t, int32_t);
int DestroyCudssBlockTridiWorkspace(turbompc::FfiDeviceAllocator&, std::uint64_t);
const char* CudssBlockTridiLastError();

using U32Buffer = ffi::Buffer<ffi::DataType::U32>;

static ffi::Error Create(cudaStream_t stream, ffi::Result<U32Buffer> result) {
  const std::uint64_t handle = CreateCudssBlockTridiWorkspace();
  if (!handle) return ffi::Error::Internal(CudssBlockTridiLastError());
  const cudaError_t error = cudaMemcpyAsync(result->typed_data(), &handle, sizeof(handle), cudaMemcpyHostToDevice, stream);
  return error == cudaSuccess ? ffi::Error::Success()
                              : ffi::Error::Internal(cudaGetErrorString(error));
}

static ffi::Error EnsureCapacity(
    turbompc::FfiDeviceAllocator allocator, cudaStream_t stream,
    std::int64_t workspace_handle, std::int64_t batch_size, std::int64_t T,
    std::int64_t n, std::int64_t dtype_bytes,
    ffi::Result<U32Buffer> result) {
  const int status = EnsureCudssBlockTridiWorkspaceCapacity(allocator, workspace_handle, batch_size, T, n, dtype_bytes);
  if (status) return ffi::Error::Internal(CudssBlockTridiLastError());
  const cudaError_t error = cudaMemsetAsync(result->typed_data(), 0, sizeof(std::uint32_t), stream);
  return error == cudaSuccess ? ffi::Error::Success()
                              : ffi::Error::Internal(cudaGetErrorString(error));
}

static ffi::Error Destroy(turbompc::FfiDeviceAllocator allocator,
                          cudaStream_t stream, std::int64_t workspace_handle,
                          ffi::Result<U32Buffer> result) {
  const int status = DestroyCudssBlockTridiWorkspace(allocator, workspace_handle);
  if (status) return ffi::Error::Internal(CudssBlockTridiLastError());
  const cudaError_t error = cudaMemsetAsync(result->typed_data(), 0, sizeof(std::uint32_t), stream);
  return error == cudaSuccess ? ffi::Error::Success()
                              : ffi::Error::Internal(cudaGetErrorString(error));
}

extern "C" XLA_FFI_Error*
TurboMpcCreateCudssBlockTridiWorkspace(XLA_FFI_CallFrame* frame) {
  static auto* handler = ffi::Ffi::Bind()
                             .Ctx<ffi::PlatformStream<cudaStream_t>>()
                             .Ret<U32Buffer>()
                             .To(Create)
                             .release();
  return handler->Call(frame);
}

extern "C" XLA_FFI_Error*
TurboMpcEnsureCudssBlockTridiWorkspaceCapacity(XLA_FFI_CallFrame* frame) {
  static auto* handler =
      ffi::Ffi::Bind()
          .Ctx<turbompc::FfiDeviceAllocator>()
          .Ctx<ffi::PlatformStream<cudaStream_t>>()
          .Attr<std::int64_t>("workspace_handle")
          .Attr<std::int64_t>("batch_size")
          .Attr<std::int64_t>("T")
          .Attr<std::int64_t>("n")
          .Attr<std::int64_t>("dtype_bytes")
          .Ret<U32Buffer>()
          .To(EnsureCapacity)
          .release();
  return handler->Call(frame);
}

extern "C" XLA_FFI_Error*
TurboMpcDestroyCudssBlockTridiWorkspace(XLA_FFI_CallFrame* frame) {
  static auto* handler = ffi::Ffi::Bind()
                             .Ctx<turbompc::FfiDeviceAllocator>()
                             .Ctx<ffi::PlatformStream<cudaStream_t>>()
                             .Attr<std::int64_t>("workspace_handle")
                             .Ret<U32Buffer>()
                             .To(Destroy)
                             .release();
  return handler->Call(frame);
}
