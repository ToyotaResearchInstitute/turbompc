#include "ffi_device_memory.h"
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "cudss_blktridi.cuh"

namespace ffi = xla::ffi;

using BufF32 = ffi::Buffer<ffi::DataType::F32>;
using BufF64 = ffi::Buffer<ffi::DataType::F64>;

template<typename BufT, typename Scalar>
static ffi::Error CudssBlkTridiCudaImpl(
    turbompc::FfiDeviceAllocator allocator,
    cudaStream_t stream,
    BufT S, BufT rhs,
    ffi::Result<BufT> x_out,
    std::int64_t workspace_handle,
    bool factorize)
{
    auto Sd = S.dimensions();
    auto rd = rhs.dimensions();

    int32_t Nb = 1;
    int32_t T, n;

    if (Sd.size() == 4) {
        // Batched: [Nb, T, n, 3n]
        Nb = static_cast<int32_t>(Sd[0]);
        T  = static_cast<int32_t>(Sd[1]);
        n  = static_cast<int32_t>(Sd[2]);
        if (Sd[3] != 3 * n)
            return ffi::Error::InvalidArgument("S last dim must be 3*n");
        if (rd.size() != 3 || rd[0] != Sd[0] || rd[1] != Sd[1] || rd[2] != Sd[2])
            return ffi::Error::InvalidArgument("rhs must be [Nb,T,n]");

        auto xod = x_out->dimensions();
        if (xod.size() != 3 || xod[0] != Sd[0] || xod[1] != Sd[1] || xod[2] != Sd[2])
            return ffi::Error::InvalidArgument("x_out must be [Nb,T,n]");
    } else if (Sd.size() == 3) {
        // Unbatched: [T, n, 3n]
        T = static_cast<int32_t>(Sd[0]);
        n = static_cast<int32_t>(Sd[1]);
        if (Sd[2] != 3 * n)
            return ffi::Error::InvalidArgument("S last dim must be 3*n");
        if (rd.size() != 2 || rd[0] != T || rd[1] != n)
            return ffi::Error::InvalidArgument("rhs must be [T,n]");

        auto xod = x_out->dimensions();
        if (xod.size() != 2 || xod[0] != T || xod[1] != n)
            return ffi::Error::InvalidArgument("x_out must be [T,n]");
    } else {
        return ffi::Error::InvalidArgument("S must be rank-3 [T,n,3n] or rank-4 [Nb,T,n,3n]");
    }

    try {
        if constexpr (std::is_same_v<Scalar, float>) {
            CudssBlkTridiSolveF32(
                allocator,
                static_cast<std::uint64_t>(workspace_handle),
                stream,
                x_out->typed_data(),
                S.typed_data(),
                rhs.typed_data(),
                T, n, factorize, Nb);
        } else {
            CudssBlkTridiSolveF64(
                allocator,
                static_cast<std::uint64_t>(workspace_handle),
                stream,
                x_out->typed_data(),
                S.typed_data(),
                rhs.typed_data(),
                T, n, factorize, Nb);
        }
    } catch (const std::invalid_argument& error) {
        return ffi::Error::InvalidArgument(error.what());
    } catch (const std::exception& error) {
        return ffi::Error::Internal(error.what());
    }

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(err));

    return ffi::Error::Success();
}

static ffi::Error CudssBlkTridiCudaFactorizeF32Handler(
    turbompc::FfiDeviceAllocator allocator, cudaStream_t stream, BufF32 S, BufF32 rhs, ffi::Result<BufF32> x_out,
    std::int64_t workspace_handle) {
    return CudssBlkTridiCudaImpl<BufF32, float>(allocator, stream, S, rhs, x_out, workspace_handle, true);
}

static ffi::Error CudssBlkTridiCudaFactorizeF64Handler(
    turbompc::FfiDeviceAllocator allocator, cudaStream_t stream, BufF64 S, BufF64 rhs, ffi::Result<BufF64> x_out,
    std::int64_t workspace_handle) {
    return CudssBlkTridiCudaImpl<BufF64, double>(allocator, stream, S, rhs, x_out, workspace_handle, true);
}

static ffi::Error CudssBlkTridiCudaSolveOnlyF32Handler(
    turbompc::FfiDeviceAllocator allocator, cudaStream_t stream, BufF32 S, BufF32 rhs, ffi::Result<BufF32> x_out,
    std::int64_t workspace_handle) {
    return CudssBlkTridiCudaImpl<BufF32, float>(allocator, stream, S, rhs, x_out, workspace_handle, false);
}

static ffi::Error CudssBlkTridiCudaSolveOnlyF64Handler(
    turbompc::FfiDeviceAllocator allocator, cudaStream_t stream, BufF64 S, BufF64 rhs, ffi::Result<BufF64> x_out,
    std::int64_t workspace_handle) {
    return CudssBlkTridiCudaImpl<BufF64, double>(allocator, stream, S, rhs, x_out, workspace_handle, false);
}

extern "C" XLA_FFI_Error* CudssBlkTridiCuda(XLA_FFI_CallFrame* call_frame) {
    static auto* handler =
        ffi::Ffi::Bind()
            .Ctx<turbompc::FfiDeviceAllocator>()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()
            .Arg<BufF32>()   // S
            .Arg<BufF32>()   // rhs
            .Ret<BufF32>()   // x_out
            .Attr<std::int64_t>("workspace_handle")
            .To(CudssBlkTridiCudaFactorizeF32Handler)
            .release();
    return handler->Call(call_frame);
}

extern "C" XLA_FFI_Error* CudssBlkTridiCudaF64(XLA_FFI_CallFrame* call_frame) {
    static auto* handler =
        ffi::Ffi::Bind()
            .Ctx<turbompc::FfiDeviceAllocator>()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()
            .Arg<BufF64>()   // S
            .Arg<BufF64>()   // rhs
            .Ret<BufF64>()   // x_out
            .Attr<std::int64_t>("workspace_handle")
            .To(CudssBlkTridiCudaFactorizeF64Handler)
            .release();
    return handler->Call(call_frame);
}

extern "C" XLA_FFI_Error* CudssBlkTridiCudaSolveOnly(XLA_FFI_CallFrame* call_frame) {
    static auto* handler =
        ffi::Ffi::Bind()
            .Ctx<turbompc::FfiDeviceAllocator>()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()
            .Arg<BufF32>()   // S
            .Arg<BufF32>()   // rhs
            .Ret<BufF32>()   // x_out
            .Attr<std::int64_t>("workspace_handle")
            .To(CudssBlkTridiCudaSolveOnlyF32Handler)
            .release();
    return handler->Call(call_frame);
}

extern "C" XLA_FFI_Error* CudssBlkTridiCudaSolveOnlyF64(XLA_FFI_CallFrame* call_frame) {
    static auto* handler =
        ffi::Ffi::Bind()
            .Ctx<turbompc::FfiDeviceAllocator>()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()
            .Arg<BufF64>()   // S
            .Arg<BufF64>()   // rhs
            .Ret<BufF64>()   // x_out
            .Attr<std::int64_t>("workspace_handle")
            .To(CudssBlkTridiCudaSolveOnlyF64Handler)
            .release();
    return handler->Call(call_frame);
}
