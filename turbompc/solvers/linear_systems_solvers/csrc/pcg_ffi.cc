#include <cstdint>
#include <string>
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "pcg_blktridi.cuh"

namespace ffi = xla::ffi;

using BufF32 = ffi::Buffer<ffi::DataType::F32>;
using BufF64 = ffi::Buffer<ffi::DataType::F64>;
using BufU32 = ffi::Buffer<ffi::DataType::U32>;

template<typename BufT, typename Scalar>
static ffi::Error PcgBlkTridiCudaImpl(
    cudaStream_t stream,
    BufT S, BufT Phiinv, BufT rhs, BufT x0,
    int64_t max_iters64,
    double eps_d,
    ffi::Result<BufT> x_out,
    ffi::Result<BufU32> iterations)
{
    auto Sd = S.dimensions();
    auto Pd = Phiinv.dimensions();
    auto rd = rhs.dimensions();
    auto x0d = x0.dimensions();

    // Support both unbatched [T,n,3n] and batched [Nb,T,n,3n]
    int32_t Nb = 1;
    int32_t T, n, cols;

    if (Sd.size() == 4) {
        // Batched: [Nb, T, n, 3n]
        Nb   = static_cast<int32_t>(Sd[0]);
        T    = static_cast<int32_t>(Sd[1]);
        n    = static_cast<int32_t>(Sd[2]);
        cols = static_cast<int32_t>(Sd[3]);

        if (cols != 3 * n)
            return ffi::Error::InvalidArgument("S last dim must be 3*n");
        if (Pd.size() != 4 || Pd[0] != Sd[0] || Pd[1] != Sd[1] || Pd[2] != Sd[2] || Pd[3] != Sd[3])
            return ffi::Error::InvalidArgument("Phiinv must match S shape [Nb,T,n,3n]");
        if (rd.size() != 3 || rd[0] != Sd[0] || rd[1] != Sd[1] || rd[2] != Sd[2])
            return ffi::Error::InvalidArgument("rhs must be [Nb,T,n]");
        if (x0d.size() != 3 || x0d[0] != Sd[0] || x0d[1] != Sd[1] || x0d[2] != Sd[2])
            return ffi::Error::InvalidArgument("x0 must be [Nb,T,n]");

        auto xod = x_out->dimensions();
        auto itd = iterations->dimensions();
        if (xod.size() != 3 || xod[0] != Sd[0] || xod[1] != Sd[1] || xod[2] != Sd[2])
            return ffi::Error::InvalidArgument("x_out must be [Nb,T,n]");
        // Accept [Nb] (direct batch) or [Nb,1] (vmap-broadcasted unbatched)
        bool it_ok = (itd.size() == 1 && itd[0] == Nb) ||
                     (itd.size() == 2 && itd[0] == Nb && itd[1] == 1);
        if (!it_ok)
            return ffi::Error::InvalidArgument("iterations must be [Nb] or [Nb,1]");
    } else if (Sd.size() == 3) {
        // Unbatched: [T, n, 3n]
        T    = static_cast<int32_t>(Sd[0]);
        n    = static_cast<int32_t>(Sd[1]);
        cols = static_cast<int32_t>(Sd[2]);

        if (cols != 3 * n)
            return ffi::Error::InvalidArgument("S last dim must be 3*n");
        if (Pd.size() != 3 || Pd[0] != Sd[0] || Pd[1] != Sd[1] || Pd[2] != Sd[2])
            return ffi::Error::InvalidArgument("Phiinv must match S shape [T,n,3n]");
        if (rd.size() != 2 || rd[0] != Sd[0] || rd[1] != Sd[1])
            return ffi::Error::InvalidArgument("rhs must be [T,n]");
        if (x0d.size() != 2 || x0d[0] != Sd[0] || x0d[1] != Sd[1])
            return ffi::Error::InvalidArgument("x0 must be [T,n]");

        auto xod = x_out->dimensions();
        auto itd = iterations->dimensions();
        if (xod.size() != 2 || xod[0] != Sd[0] || xod[1] != Sd[1])
            return ffi::Error::InvalidArgument("x_out must be [T,n]");
        if (itd.size() != 1 || itd[0] != 1)
            return ffi::Error::InvalidArgument("iterations must be [1]");
    } else {
        return ffi::Error::InvalidArgument("S must be rank-3 [T,n,3n] or rank-4 [Nb,T,n,3n]");
    }

    int32_t max_iters = static_cast<int32_t>(max_iters64);
    Scalar eps = static_cast<Scalar>(eps_d);

    if constexpr (std::is_same_v<Scalar, float>) {
        LaunchPcgBlkTridiF32(
            stream,
            x_out->typed_data(),
            iterations->typed_data(),
            S.typed_data(),
            Phiinv.typed_data(),
            rhs.typed_data(),
            x0.typed_data(),
            eps,
            max_iters, T, n, Nb);
    } else {
        LaunchPcgBlkTridiF64(
            stream,
            x_out->typed_data(),
            iterations->typed_data(),
            S.typed_data(),
            Phiinv.typed_data(),
            rhs.typed_data(),
            x0.typed_data(),
            eps,
            max_iters, T, n, Nb);
    }

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(err));

    return ffi::Error::Success();
}

extern "C" XLA_FFI_Error* PcgBlkTridiCuda(XLA_FFI_CallFrame* call_frame) {
    static auto* handler =
        ffi::Ffi::Bind()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()
            .Arg<BufF32>()   // S
            .Arg<BufF32>()   // Phiinv
            .Arg<BufF32>()   // rhs
            .Arg<BufF32>()   // x0
            .Attr<int64_t>("max_iters")
            .Attr<double>("eps")
            .Ret<BufF32>()   // x_out
            .Ret<BufU32>()   // iterations
            .To(PcgBlkTridiCudaImpl<BufF32, float>)
            .release();
    return handler->Call(call_frame);
}

extern "C" XLA_FFI_Error* PcgBlkTridiCudaF64(XLA_FFI_CallFrame* call_frame) {
    static auto* handler =
        ffi::Ffi::Bind()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()
            .Arg<BufF64>()   // S
            .Arg<BufF64>()   // Phiinv
            .Arg<BufF64>()   // rhs
            .Arg<BufF64>()   // x0
            .Attr<int64_t>("max_iters")
            .Attr<double>("eps")
            .Ret<BufF64>()   // x_out
            .Ret<BufU32>()   // iterations
            .To(PcgBlkTridiCudaImpl<BufF64, double>)
            .release();
    return handler->Call(call_frame);
}
