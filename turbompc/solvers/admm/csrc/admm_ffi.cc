#include <cstdint>
#include <string>
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "admm_fused.cuh"

namespace ffi = xla::ffi;

using BufF32 = ffi::Buffer<ffi::DataType::F32>;
using BufF64 = ffi::Buffer<ffi::DataType::F64>;
using BufU32 = ffi::Buffer<ffi::DataType::U32>;

template<typename BufT, typename Scalar>
static ffi::Error AdmmFusedCudaImpl(
    cudaStream_t stream,
    // QP data
    BufT S, BufT Phiinv, BufT D, BufT E, BufT q,
    BufT A0, BufT A_minus, BufT A_plus,
    BufT G, BufT l_bounds, BufT u_bounds,
    BufT c0, BufT c_dyn,
    // Warm-start
    BufT x0, BufT z_g0, BufT y_g0,
    BufT y_f_0_init, BufT y_f_dyn_init,
    BufT xi_g0, BufT rho_bar_init,
    BufT slack_weight_init,  // [Nb] runtime slack weight (JAX-traceable)
    // Config attrs
    int64_t max_iter, int64_t pcg_max_iter, int64_t check_every,
    double eps_abs, double eps_rel, double sigma,
    double rho_f_factor, double alpha, double pcg_eps,
    int64_t nx64, int64_t n064, int64_t m64,
    int64_t use_slack64,
    int64_t adapt_rho_every64, double adaptive_rho_tolerance,
    double rho_min, double rho_max,
    // Results (10 + 2 workspace)
    ffi::Result<BufT>  x_out,
    ffi::Result<BufU32> iters_out,
    ffi::Result<BufT>  x_blocks_out,
    ffi::Result<BufT>  z_g_out,
    ffi::Result<BufT>  y_g_out,
    ffi::Result<BufT>  y_f_0_out,
    ffi::Result<BufT>  y_f_dyn_out,
    ffi::Result<BufT>  xi_g_out,
    ffi::Result<BufT>  rho_bar_out,
    ffi::Result<BufT>  kernel_ns_out,
    ffi::Result<BufT>  S_work_out,
    ffi::Result<BufT>  Phiinv_work_out)
{
    // Extract dimensions from S shape: [Nb, T, n, 3n] or [T, n, 3n] (Nb=1)
    auto Sd = S.dimensions();
    int32_t Nb, T, n;
    if (Sd.size() == 4) {
        Nb = static_cast<int32_t>(Sd[0]);
        T  = static_cast<int32_t>(Sd[1]);
        n  = static_cast<int32_t>(Sd[2]);
    } else if (Sd.size() == 3) {
        Nb = 1;
        T  = static_cast<int32_t>(Sd[0]);
        n  = static_cast<int32_t>(Sd[1]);
    } else {
        return ffi::Error::InvalidArgument("S must be rank-3 [T, n, 3n] or rank-4 [Nb, T, n, 3n]");
    }
    int32_t nx = static_cast<int32_t>(nx64);
    int32_t n0 = static_cast<int32_t>(n064);
    int32_t m  = static_cast<int32_t>(m64);

    if (Sd[Sd.size() - 1] != 3 * n)
        return ffi::Error::InvalidArgument("S last dim must be 3*n");

    // Build config
    ADMMConfig cfg;
    cfg.max_iter     = static_cast<int32_t>(max_iter);
    cfg.pcg_max_iter = static_cast<int32_t>(pcg_max_iter);
    cfg.check_every  = static_cast<int32_t>(check_every);
    cfg.eps_abs      = eps_abs;
    cfg.eps_rel      = eps_rel;
    cfg.sigma        = sigma;
    cfg.rho_f_factor = rho_f_factor;
    cfg.alpha        = alpha;
    cfg.pcg_eps      = pcg_eps;
    cfg.adapt_rho    = (adapt_rho_every64 > 0);
    cfg.use_slack    = (use_slack64 != 0);
    cfg.adapt_rho_every        = static_cast<int32_t>(adapt_rho_every64);
    cfg.adaptive_rho_tolerance = adaptive_rho_tolerance;
    cfg.rho_min                = rho_min;
    cfg.rho_max                = rho_max;

    cudaError_t launch_err;
    if constexpr (std::is_same_v<Scalar, float>) {
        launch_err = LaunchADMMFusedF32(
            stream,
            x_out->typed_data(),
            iters_out->typed_data(),
            x_blocks_out->typed_data(),
            z_g_out->typed_data(),
            y_g_out->typed_data(),
            y_f_0_out->typed_data(),
            y_f_dyn_out->typed_data(),
            xi_g_out->typed_data(),
            rho_bar_out->typed_data(),
            kernel_ns_out->typed_data(),
            S_work_out->typed_data(),
            Phiinv_work_out->typed_data(),
            S.typed_data(), Phiinv.typed_data(),
            D.typed_data(), E.typed_data(), q.typed_data(),
            A0.typed_data(), A_minus.typed_data(), A_plus.typed_data(),
            G.typed_data(), l_bounds.typed_data(), u_bounds.typed_data(),
            c0.typed_data(), c_dyn.typed_data(),
            x0.typed_data(), z_g0.typed_data(), y_g0.typed_data(),
            y_f_0_init.typed_data(), y_f_dyn_init.typed_data(),
            xi_g0.typed_data(), rho_bar_init.typed_data(),
            slack_weight_init.typed_data(),
            T, n, nx, n0, m, Nb, cfg);
    } else {
        launch_err = LaunchADMMFusedF64(
            stream,
            x_out->typed_data(),
            iters_out->typed_data(),
            x_blocks_out->typed_data(),
            z_g_out->typed_data(),
            y_g_out->typed_data(),
            y_f_0_out->typed_data(),
            y_f_dyn_out->typed_data(),
            xi_g_out->typed_data(),
            rho_bar_out->typed_data(),
            kernel_ns_out->typed_data(),
            S_work_out->typed_data(),
            Phiinv_work_out->typed_data(),
            S.typed_data(), Phiinv.typed_data(),
            D.typed_data(), E.typed_data(), q.typed_data(),
            A0.typed_data(), A_minus.typed_data(), A_plus.typed_data(),
            G.typed_data(), l_bounds.typed_data(), u_bounds.typed_data(),
            c0.typed_data(), c_dyn.typed_data(),
            x0.typed_data(), z_g0.typed_data(), y_g0.typed_data(),
            y_f_0_init.typed_data(), y_f_dyn_init.typed_data(),
            xi_g0.typed_data(), rho_bar_init.typed_data(),
            slack_weight_init.typed_data(),
            T, n, nx, n0, m, Nb, cfg);
    }

    if (launch_err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(launch_err));

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(err));

    return ffi::Error::Success();
}

extern "C" XLA_FFI_Error* AdmmFusedCuda(XLA_FFI_CallFrame* call_frame) {
    static auto* handler =
        ffi::Ffi::Bind()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()
            // QP data (13 args)
            .Arg<BufF32>()   // S
            .Arg<BufF32>()   // Phiinv
            .Arg<BufF32>()   // D
            .Arg<BufF32>()   // E
            .Arg<BufF32>()   // q
            .Arg<BufF32>()   // A0
            .Arg<BufF32>()   // A_minus
            .Arg<BufF32>()   // A_plus
            .Arg<BufF32>()   // G
            .Arg<BufF32>()   // l_bounds
            .Arg<BufF32>()   // u_bounds
            .Arg<BufF32>()   // c0
            .Arg<BufF32>()   // c_dyn
            // Warm-start (7 args)
            .Arg<BufF32>()   // x0
            .Arg<BufF32>()   // z_g0
            .Arg<BufF32>()   // y_g0
            .Arg<BufF32>()   // y_f_0_init
            .Arg<BufF32>()   // y_f_dyn_init
            .Arg<BufF32>()   // xi_g0
            .Arg<BufF32>()   // rho_bar_init
            .Arg<BufF32>()   // slack_weight_init
            // Config attrs (16)
            .Attr<int64_t>("max_iter")
            .Attr<int64_t>("pcg_max_iter")
            .Attr<int64_t>("check_every")
            .Attr<double>("eps_abs")
            .Attr<double>("eps_rel")
            .Attr<double>("sigma")
            .Attr<double>("rho_f_factor")
            .Attr<double>("alpha")
            .Attr<double>("pcg_eps")
            .Attr<int64_t>("nx")
            .Attr<int64_t>("n0")
            .Attr<int64_t>("m")
            .Attr<int64_t>("use_slack")
            .Attr<int64_t>("adapt_rho_every")
            .Attr<double>("adaptive_rho_tolerance")
            .Attr<double>("rho_min")
            .Attr<double>("rho_max")
            // Results (10 + 2 workspace)
            .Ret<BufF32>()   // x_out
            .Ret<BufU32>()   // iters_out
            .Ret<BufF32>()   // x_blocks_out
            .Ret<BufF32>()   // z_g_out
            .Ret<BufF32>()   // y_g_out
            .Ret<BufF32>()   // y_f_0_out
            .Ret<BufF32>()   // y_f_dyn_out
            .Ret<BufF32>()   // xi_g_out
            .Ret<BufF32>()   // rho_bar_out
            .Ret<BufF32>()   // kernel_ns_out
            .Ret<BufF32>()   // S_work
            .Ret<BufF32>()   // Phiinv_work
            .To(AdmmFusedCudaImpl<BufF32, float>)
            .release();
    return handler->Call(call_frame);
}

extern "C" XLA_FFI_Error* AdmmFusedCudaF64(XLA_FFI_CallFrame* call_frame) {
    static auto* handler =
        ffi::Ffi::Bind()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()
            // QP data (13 args)
            .Arg<BufF64>()   // S
            .Arg<BufF64>()   // Phiinv
            .Arg<BufF64>()   // D
            .Arg<BufF64>()   // E
            .Arg<BufF64>()   // q
            .Arg<BufF64>()   // A0
            .Arg<BufF64>()   // A_minus
            .Arg<BufF64>()   // A_plus
            .Arg<BufF64>()   // G
            .Arg<BufF64>()   // l_bounds
            .Arg<BufF64>()   // u_bounds
            .Arg<BufF64>()   // c0
            .Arg<BufF64>()   // c_dyn
            // Warm-start (7 args)
            .Arg<BufF64>()   // x0
            .Arg<BufF64>()   // z_g0
            .Arg<BufF64>()   // y_g0
            .Arg<BufF64>()   // y_f_0_init
            .Arg<BufF64>()   // y_f_dyn_init
            .Arg<BufF64>()   // xi_g0
            .Arg<BufF64>()   // rho_bar_init
            .Arg<BufF64>()   // slack_weight_init
            // Config attrs (16)
            .Attr<int64_t>("max_iter")
            .Attr<int64_t>("pcg_max_iter")
            .Attr<int64_t>("check_every")
            .Attr<double>("eps_abs")
            .Attr<double>("eps_rel")
            .Attr<double>("sigma")
            .Attr<double>("rho_f_factor")
            .Attr<double>("alpha")
            .Attr<double>("pcg_eps")
            .Attr<int64_t>("nx")
            .Attr<int64_t>("n0")
            .Attr<int64_t>("m")
            .Attr<int64_t>("use_slack")
            .Attr<int64_t>("adapt_rho_every")
            .Attr<double>("adaptive_rho_tolerance")
            .Attr<double>("rho_min")
            .Attr<double>("rho_max")
            // Results (10 + 2 workspace)
            .Ret<BufF64>()   // x_out
            .Ret<BufU32>()   // iters_out
            .Ret<BufF64>()   // x_blocks_out
            .Ret<BufF64>()   // z_g_out
            .Ret<BufF64>()   // y_g_out
            .Ret<BufF64>()   // y_f_0_out
            .Ret<BufF64>()   // y_f_dyn_out
            .Ret<BufF64>()   // xi_g_out
            .Ret<BufF64>()   // rho_bar_out
            .Ret<BufF64>()   // kernel_ns_out
            .Ret<BufF64>()   // S_work
            .Ret<BufF64>()   // Phiinv_work
            .To(AdmmFusedCudaImpl<BufF64, double>)
            .release();
    return handler->Call(call_frame);
}
