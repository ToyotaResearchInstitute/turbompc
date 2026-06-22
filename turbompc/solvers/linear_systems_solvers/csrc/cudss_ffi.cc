#include <cstdint>
#include <string>
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "cudss_blktridi.cuh"

namespace ffi = xla::ffi;

using BufF32 = ffi::Buffer<ffi::DataType::F32>;
using BufF64 = ffi::Buffer<ffi::DataType::F64>;

template<typename BufT, typename Scalar>
static ffi::Error CudssBlkTridiCudaImpl(
    cudaStream_t stream,
    BufT S, BufT rhs,
    ffi::Result<BufT> x_out)
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

    if constexpr (std::is_same_v<Scalar, float>) {
        CudssBlkTridiSolveF32(
            stream,
            x_out->typed_data(),
            S.typed_data(),
            rhs.typed_data(),
            T, n, Nb);
    } else {
        CudssBlkTridiSolveF64(
            stream,
            x_out->typed_data(),
            S.typed_data(),
            rhs.typed_data(),
            T, n, Nb);
    }

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(err));

    return ffi::Error::Success();
}

extern "C" XLA_FFI_Error* CudssBlkTridiCuda(XLA_FFI_CallFrame* call_frame) {
    static auto* handler =
        ffi::Ffi::Bind()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()
            .Arg<BufF32>()   // S
            .Arg<BufF32>()   // rhs
            .Ret<BufF32>()   // x_out
            .To(CudssBlkTridiCudaImpl<BufF32, float>)
            .release();
    return handler->Call(call_frame);
}

extern "C" XLA_FFI_Error* CudssBlkTridiCudaF64(XLA_FFI_CallFrame* call_frame) {
    static auto* handler =
        ffi::Ffi::Bind()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()
            .Arg<BufF64>()   // S
            .Arg<BufF64>()   // rhs
            .Ret<BufF64>()   // x_out
            .To(CudssBlkTridiCudaImpl<BufF64, double>)
            .release();
    return handler->Call(call_frame);
}
