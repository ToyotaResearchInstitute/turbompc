#include "cudss_sparse_kkt.cuh"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cstdint>
#include <string>

namespace ffi = xla::ffi;

using BufI32 = ffi::Buffer<ffi::DataType::S32>;
using BufF32 = ffi::Buffer<ffi::DataType::F32>;
using BufF64 = ffi::Buffer<ffi::DataType::F64>;

// FFI handler for f32/f64 sparse KKT solve
template<typename ScalarBuf, typename Scalar>
static ffi::Error CudssSparseKktImpl(
    cudaStream_t stream,
    BufI32 rowPtr,
    BufI32 colIdx,
    ScalarBuf values,
    ScalarBuf rhs,
    ffi::Result<ScalarBuf> sol
) {
    auto rowPtr_dims = rowPtr.dimensions();
    auto colIdx_dims = colIdx.dimensions();
    auto values_dims = values.dimensions();
    auto rhs_dims = rhs.dimensions();
    auto sol_dims = sol->dimensions();

    // Infer n from rowPtr dimensions
    if (rowPtr_dims.size() != 1 || rowPtr_dims[0] < 2) {
        return ffi::Error::InvalidArgument("rowPtr must be 1D with size >= 2");
    }
    int32_t n = rowPtr_dims[0] - 1;

    // Validate dimensions
    if (values_dims.size() != 1) {
        return ffi::Error::InvalidArgument("values must be 1D");
    }
    if (colIdx_dims.size() != 1) {
        return ffi::Error::InvalidArgument("colIdx must be 1D");
    }
    if (rhs_dims.size() != 1 || rhs_dims[0] != n) {
        return ffi::Error::InvalidArgument("rhs must have size n");
    }
    if (sol_dims.size() != 1 || sol_dims[0] != n) {
        return ffi::Error::InvalidArgument("sol must have size n");
    }

    // nnz is the exact size of colIdx (caller allocates colIdx with exactly nnz
    // elements — no padding, no device→host sync needed).
    int32_t nnz_host = static_cast<int32_t>(colIdx_dims[0]);

    if (values_dims[0] != nnz_host) {
        return ffi::Error::InvalidArgument("values size must equal colIdx size");
    }

    // Call CUDA kernel (device pointers) — cuDSS/CUDA errors throw and are
    // converted to a catchable FFI Internal error (no process abort()).
    try {
        if constexpr (std::is_same_v<Scalar, float>) {
            turbompc::cudss_sparse_kkt::solve_sparse_kkt_f32(
                stream,
                rowPtr.typed_data(),
                colIdx.typed_data(),
                values.typed_data(),
                rhs.typed_data(),
                sol->typed_data(),
                n,
                nnz_host
            );
        } else {
            turbompc::cudss_sparse_kkt::solve_sparse_kkt_f64(
                stream,
                rowPtr.typed_data(),
                colIdx.typed_data(),
                values.typed_data(),
                rhs.typed_data(),
                sol->typed_data(),
                n,
                nnz_host
            );
        }
    } catch (const std::exception& e) {
        return ffi::Error::Internal(std::string("cudss_sparse_kkt: ") + e.what());
    }

    return ffi::Error::Success();
}

// C API entry points
extern "C" XLA_FFI_Error* CudssSparseKktF32(XLA_FFI_CallFrame* call_frame) {
    static auto* handler =
        ffi::Ffi::Bind()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()  // CUDA stream
            .Arg<BufI32>()   // rowPtr
            .Arg<BufI32>()   // colIdx
            .Arg<BufF32>()   // values
            .Arg<BufF32>()   // rhs
            .Ret<BufF32>()   // sol
            .To(CudssSparseKktImpl<BufF32, float>)
            .release();
    return handler->Call(call_frame);
}

extern "C" XLA_FFI_Error* CudssSparseKktF64(XLA_FFI_CallFrame* call_frame) {
    static auto* handler =
        ffi::Ffi::Bind()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()  // CUDA stream
            .Arg<BufI32>()   // rowPtr
            .Arg<BufI32>()   // colIdx
            .Arg<BufF64>()   // values
            .Arg<BufF64>()   // rhs
            .Ret<BufF64>()   // sol
            .To(CudssSparseKktImpl<BufF64, double>)
            .release();
    return handler->Call(call_frame);
}

// ---------------------------------------------------------------------------
// Dense-input FFI handlers: take n×n dense KKT matrix + rhs, convert to CSR
// on GPU (cuSPARSE) and solve with cuDSS — no host round-trips.
// ---------------------------------------------------------------------------
template<typename ScalarBuf, typename Scalar>
static ffi::Error CudssSparseKktDenseImpl(
    cudaStream_t stream,
    ScalarBuf kkt,
    ScalarBuf rhs,
    ffi::Result<ScalarBuf> sol
) {
    auto kkt_dims = kkt.dimensions();
    auto rhs_dims = rhs.dimensions();
    auto sol_dims = sol->dimensions();

    if (kkt_dims.size() != 2 || kkt_dims[0] != kkt_dims[1]) {
        return ffi::Error::InvalidArgument("kkt must be a square 2D matrix");
    }
    int32_t n = static_cast<int32_t>(kkt_dims[0]);

    if (rhs_dims.size() != 1 || static_cast<int32_t>(rhs_dims[0]) != n) {
        return ffi::Error::InvalidArgument("rhs must be 1D with size n");
    }
    if (sol_dims.size() != 1 || static_cast<int32_t>(sol_dims[0]) != n) {
        return ffi::Error::InvalidArgument("sol must be 1D with size n");
    }

    // Call CUDA kernel (device pointers) — cuDSS/CUDA errors throw and are
    // converted to a catchable FFI Internal error (no process abort()).
    try {
        if constexpr (std::is_same_v<Scalar, float>) {
            turbompc::cudss_sparse_kkt::solve_sparse_kkt_from_dense_f32(
                stream, kkt.typed_data(), rhs.typed_data(), sol->typed_data(), n);
        } else {
            turbompc::cudss_sparse_kkt::solve_sparse_kkt_from_dense_f64(
                stream, kkt.typed_data(), rhs.typed_data(), sol->typed_data(), n);
        }
    } catch (const std::exception& e) {
        return ffi::Error::Internal(std::string("cudss_sparse_kkt_dense: ") + e.what());
    }
    return ffi::Error::Success();
}

extern "C" XLA_FFI_Error* CudssSparseKktDenseF32(XLA_FFI_CallFrame* call_frame) {
    static auto* handler =
        ffi::Ffi::Bind()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()
            .Arg<BufF32>()   // kkt (n×n dense)
            .Arg<BufF32>()   // rhs (n)
            .Ret<BufF32>()   // sol (n)
            .To(CudssSparseKktDenseImpl<BufF32, float>)
            .release();
    return handler->Call(call_frame);
}

extern "C" XLA_FFI_Error* CudssSparseKktDenseF64(XLA_FFI_CallFrame* call_frame) {
    static auto* handler =
        ffi::Ffi::Bind()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()
            .Arg<BufF64>()   // kkt (n×n dense)
            .Arg<BufF64>()   // rhs (n)
            .Ret<BufF64>()   // sol (n)
            .To(CudssSparseKktDenseImpl<BufF64, double>)
            .release();
    return handler->Call(call_frame);
}
