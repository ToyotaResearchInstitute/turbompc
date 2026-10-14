#include "ffi_device_memory.h"
#include "cudss_sparse_kkt.cuh"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cstdint>
#include <string>

namespace ffi = xla::ffi;

uint64_t CreateCudssSparseKktWorkspace(const int32_t*, const int32_t*, int32_t, int32_t, int32_t);
int DestroyCudssSparseKktWorkspace(turbompc::FfiDeviceAllocator&, uint64_t);
int PrepareCudssSparseKktWorkspace(turbompc::FfiDeviceAllocator&, uint64_t, bool, int64_t*);
const char* CudssSparseKktLastError();

static ffi::Error CreateWorkspace(
    turbompc::FfiDeviceAllocator allocator,
    cudaStream_t stream,
    ffi::Span<const int32_t> row_ptr, ffi::Span<const int32_t> col_idx,
    int64_t dtype_bytes, ffi::Result<ffi::Buffer<ffi::DataType::U32>> out) {
    if (row_ptr.size() < 2 || col_idx.size() < 1)
        return ffi::Error::InvalidArgument("invalid sparse KKT pattern");
    const uint64_t handle = CreateCudssSparseKktWorkspace(
        &row_ptr[0], &col_idx[0], row_ptr.size() - 1, col_idx.size(), dtype_bytes);
    if (!handle) return ffi::Error::Internal(CudssSparseKktLastError());
    int64_t unused[2] = {};
    if (PrepareCudssSparseKktWorkspace(allocator, handle, false, unused)) {
        const std::string message = CudssSparseKktLastError();
        cudaDeviceSynchronize();
        DestroyCudssSparseKktWorkspace(allocator, handle);
        return ffi::Error::Internal(message);
    }
    const auto error = cudaMemcpyAsync(out->typed_data(), &handle, sizeof(handle),
                                       cudaMemcpyHostToDevice, stream);
    return error == cudaSuccess ? ffi::Error::Success()
                               : ffi::Error::Internal(cudaGetErrorString(error));
}

extern "C" XLA_FFI_Error*
TurboMpcCreateCudssSparseKktWorkspace(XLA_FFI_CallFrame* frame) {
    static auto* handler = ffi::Ffi::Bind()
        .Ctx<turbompc::FfiDeviceAllocator>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<ffi::Span<const int32_t>>("row_ptr")
        .Attr<ffi::Span<const int32_t>>("col_idx")
        .Attr<int64_t>("dtype_bytes")
        .Ret<ffi::Buffer<ffi::DataType::U32>>()
        .To(CreateWorkspace)
        .release();
    return handler->Call(frame);
}

static ffi::Error DestroyWorkspace(
    turbompc::FfiDeviceAllocator allocator, cudaStream_t stream,
    int64_t workspace_handle,
    ffi::Result<ffi::Buffer<ffi::DataType::U32>> done) {
    const int status = DestroyCudssSparseKktWorkspace(allocator, workspace_handle);
    if (status) return ffi::Error::Internal(CudssSparseKktLastError());
    const auto error = cudaMemsetAsync(done->typed_data(), 0, sizeof(uint32_t), stream);
    return error == cudaSuccess ? ffi::Error::Success()
                                : ffi::Error::Internal(cudaGetErrorString(error));
}

extern "C" XLA_FFI_Error*
TurboMpcDestroyCudssSparseKktWorkspace(XLA_FFI_CallFrame* frame) {
    static auto* handler = ffi::Ffi::Bind()
        .Ctx<turbompc::FfiDeviceAllocator>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("workspace_handle")
        .Ret<ffi::Buffer<ffi::DataType::U32>>()
        .To(DestroyWorkspace)
        .release();
    return handler->Call(frame);
}

static ffi::Error Probe(turbompc::FfiDeviceAllocator allocator, cudaStream_t stream, int64_t handle,
                        ffi::Result<ffi::Buffer<ffi::DataType::U32>> out) {
    int64_t estimates[2] = {};
    if (PrepareCudssSparseKktWorkspace(allocator, handle, true, estimates))
        return ffi::Error::Internal(CudssSparseKktLastError());
    auto status = cudaMemcpyAsync(out->typed_data(), estimates, sizeof(estimates),
                                  cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    return status == cudaSuccess ? ffi::Error::Success()
                                 : ffi::Error::Internal(cudaGetErrorString(status));
}

extern "C" XLA_FFI_Error* CudssSparseKktProbe(XLA_FFI_CallFrame* frame) {
    static auto* handler = ffi::Ffi::Bind()
        .Ctx<turbompc::FfiDeviceAllocator>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("workspace_handle")
        .Ret<ffi::Buffer<ffi::DataType::U32>>()
        .To(Probe)
        .release();
    return handler->Call(frame);
}

using BufF32 = ffi::Buffer<ffi::DataType::F32>;
using BufF64 = ffi::Buffer<ffi::DataType::F64>;

// FFI handler for f32/f64 sparse KKT solve
template<typename ScalarBuf, typename Scalar>
static ffi::Error CudssSparseKktImpl(
    turbompc::FfiDeviceAllocator allocator,
    cudaStream_t stream,
    ScalarBuf values,
    ScalarBuf rhs,
    int64_t n64,
    int64_t nnz64,
    int64_t workspace_handle,
    ffi::Result<ScalarBuf> sol
) {
    auto values_dims = values.dimensions();
    auto rhs_dims = rhs.dimensions();
    auto sol_dims = sol->dimensions();

    int32_t n = static_cast<int32_t>(n64);
    int32_t nnz = static_cast<int32_t>(nnz64);

    // Validate dimensions
    if (values_dims.size() != 1) {
        return ffi::Error::InvalidArgument("values must be 1D");
    }
    if (rhs_dims.size() != 1 || rhs_dims[0] != n) {
        return ffi::Error::InvalidArgument("rhs must have size n");
    }
    if (sol_dims.size() != 1 || sol_dims[0] != n) {
        return ffi::Error::InvalidArgument("sol must have size n");
    }

    if (values_dims[0] != nnz) {
        return ffi::Error::InvalidArgument("values size must equal workspace nnz");
    }

    // Call CUDA kernel (device pointers) — cuDSS/CUDA errors throw and are
    // converted to a catchable FFI Internal error (no process abort()).
    try {
        if constexpr (std::is_same_v<Scalar, float>) {
            turbompc::cudss_sparse_kkt::solve_sparse_kkt_f32(
                allocator,
                static_cast<uint64_t>(workspace_handle),
                stream,
                values.typed_data(),
                rhs.typed_data(),
                sol->typed_data(),
                n,
                nnz
            );
        } else {
            turbompc::cudss_sparse_kkt::solve_sparse_kkt_f64(
                allocator,
                static_cast<uint64_t>(workspace_handle),
                stream,
                values.typed_data(),
                rhs.typed_data(),
                sol->typed_data(),
                n,
                nnz
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
            .Ctx<turbompc::FfiDeviceAllocator>()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()  // CUDA stream
            .Arg<BufF32>()   // values
            .Arg<BufF32>()   // rhs
            .Attr<int64_t>("n")
            .Attr<int64_t>("nnz")
            .Attr<int64_t>("workspace_handle")
            .Ret<BufF32>()   // sol
            .To(CudssSparseKktImpl<BufF32, float>)
            .release();
    return handler->Call(call_frame);
}

extern "C" XLA_FFI_Error* CudssSparseKktF64(XLA_FFI_CallFrame* call_frame) {
    static auto* handler =
        ffi::Ffi::Bind()
            .Ctx<turbompc::FfiDeviceAllocator>()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()  // CUDA stream
            .Arg<BufF64>()   // values
            .Arg<BufF64>()   // rhs
            .Attr<int64_t>("n")
            .Attr<int64_t>("nnz")
            .Attr<int64_t>("workspace_handle")
            .Ret<BufF64>()   // sol
            .To(CudssSparseKktImpl<BufF64, double>)
            .release();
    return handler->Call(call_frame);
}
