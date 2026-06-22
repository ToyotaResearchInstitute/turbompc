#include "cudss_sparse_kkt.cuh"
#include <cudss.h>
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <map>
#include <utility>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      char _buf[256];                                                          \
      snprintf(_buf, sizeof(_buf), "CUDA error at %s:%d: %s",                  \
               __FILE__, __LINE__, cudaGetErrorString(err));                   \
      throw std::runtime_error(_buf);                                          \
    }                                                                          \
  } while (0)

#define CUDSS_CHECK(call)                                                      \
  do {                                                                         \
    cudssStatus_t err = call;                                                  \
    if (err != CUDSS_STATUS_SUCCESS) {                                         \
      char _buf[256];                                                          \
      snprintf(_buf, sizeof(_buf), "cuDSS error at %s:%d: code %d",            \
               __FILE__, __LINE__, static_cast<int>(err));                     \
      throw std::runtime_error(_buf);                                          \
    }                                                                          \
  } while (0)

#define CUSPARSE_CHECK(call)                                                   \
  do {                                                                         \
    cusparseStatus_t err = call;                                               \
    if (err != CUSPARSE_STATUS_SUCCESS) {                                      \
      char _buf[256];                                                          \
      snprintf(_buf, sizeof(_buf), "cuSPARSE error at %s:%d: %s",              \
               __FILE__, __LINE__, cusparseGetErrorString(err));               \
      throw std::runtime_error(_buf);                                          \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Global cuDSS handle — created once, reused across all calls.
// ---------------------------------------------------------------------------
static cudssHandle_t g_cudss_handle = nullptr;
static std::once_flag g_cudss_init_flag;

static cudssHandle_t get_cudss_handle() {
    std::call_once(g_cudss_init_flag, []() {
        cudssCreate(&g_cudss_handle);
    });
    return g_cudss_handle;
}

// ---------------------------------------------------------------------------
// Per-pattern analysis cache.
// Key: (n, nnz, is_double).  Value: cudssData_t with completed analysis.
// The analysis phase (symbolic factorization + fill-reducing ordering) only
// depends on the sparsity pattern, not on the values.  Caching it means every
// subsequent call with the same (n, nnz) skips analysis and goes straight to
// numeric factorization + solve.
// ---------------------------------------------------------------------------
struct AnalysisCacheKey {
    int32_t n;
    int32_t nnz;
    bool    is_double;
    bool operator<(const AnalysisCacheKey& o) const {
        if (n != o.n)         return n < o.n;
        if (nnz != o.nnz)     return nnz < o.nnz;
        return is_double < o.is_double;
    }
};
static std::map<AnalysisCacheKey, cudssData_t> g_analysis_cache;
static std::mutex g_analysis_mutex;

// ---------------------------------------------------------------------------
// Cache control entry points (callable from Python via ctypes — NOT FFI).
// The analysis cache otherwise lives for the process lifetime with no
// eviction; long-lived processes that solve many distinct sparsity patterns
// (interactive sessions, shape-sweeping training) accumulate one
// cudssData_t (symbolic factorization + GPU workspace) per pattern forever.
//   ClearCudssAnalysisCacheImpl(): destroy every cached analysis object and
//                                  empty the map.
//   CudssAnalysisCacheSize():      number of live entries (debug/test helper).
// ---------------------------------------------------------------------------
extern "C" void ClearCudssAnalysisCacheImpl() {
    std::lock_guard<std::mutex> lock(g_analysis_mutex);
    cudssHandle_t handle = get_cudss_handle();
    for (auto& kv : g_analysis_cache) {
        if (kv.second) cudssDataDestroy(handle, kv.second);
    }
    g_analysis_cache.clear();
}

extern "C" int CudssAnalysisCacheSize() {
    std::lock_guard<std::mutex> lock(g_analysis_mutex);
    return static_cast<int>(g_analysis_cache.size());
}

// ---------------------------------------------------------------------------
// Eager device-memory probe (ctypes, NOT FFI; runs OUTSIDE the traced graph).
// Runs cuDSS PHASE_ANALYSIS on a SINGLE block of the given sparsity pattern
// and reports, via out[2]:
//   out[0] = device bytes consumed by that analysis (free_before - free_after)
//   out[1] = device free bytes at probe time (budget cuDSS can use)
// Block-diagonal stacks are exactly k identical blocks, so the chunk-safe
// size is floor(out[1]*safety / out[0]).  Returns 0 on success, -1 on any
// CUDA/cuDSS error (caller falls back to a conservative chunk=1).
//
// rowPtr_host (n+1) and colIdx_host (nnz) are HOST int32 arrays (the cached
// numpy CSR pattern).  Values are irrelevant to symbolic analysis; a dummy
// device buffer is used so cudssMatrixCreateCsr has a valid pointer.
// ---------------------------------------------------------------------------
extern "C" int CudssProbeDeviceBytesImpl(
    const int32_t* rowPtr_host, const int32_t* colIdx_host,
    int32_t n, int32_t nnz, int is_double, int64_t* out /* [2] */)
{
    out[0] = 0; out[1] = 0;
    int32_t *d_rowPtr = nullptr, *d_colIdx = nullptr;
    void *d_vals = nullptr, *d_rhs = nullptr, *d_sol = nullptr;
    cudssMatrix_t matA = nullptr, matX = nullptr, matB = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    int rc = 0;
    try {
        cudssHandle_t handle = get_cudss_handle();
        const cudssDataType_t vtype = is_double ? CUDSS_R_64F : CUDSS_R_32F;
        const size_t esz = is_double ? sizeof(double) : sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_rowPtr, (size_t)(n + 1) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_colIdx, (size_t)nnz * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_rowPtr, rowPtr_host,
                              (size_t)(n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colIdx, colIdx_host,
                              (size_t)nnz * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_vals, (size_t)nnz * esz));
        CUDA_CHECK(cudaMalloc(&d_rhs, (size_t)n * esz));
        CUDA_CHECK(cudaMalloc(&d_sol, (size_t)n * esz));
        CUDA_CHECK(cudaMemset(d_vals, 0, (size_t)nnz * esz));
        CUDA_CHECK(cudaMemset(d_rhs, 0, (size_t)n * esz));

        CUDSS_CHECK(cudssConfigCreate(&config));
        cudssReorderingAlg_t ralg = CUDSS_REORDERING_ALG_DEFAULT;
        CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG,
                                   &ralg, sizeof(ralg)));
        CUDSS_CHECK(cudssMatrixCreateCsr(
            &matA, n, n, nnz, d_rowPtr, nullptr, d_colIdx, d_vals,
            CUDSS_R_32I, CUDSS_R_32I, vtype, CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_FULL,
            CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(&matX, n, 1, n, d_sol, vtype,
                                        CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&matB, n, 1, n, d_rhs, vtype,
                                        CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssDataCreate(handle, &data));

        // Run analysis phase (symbolic factorization + reordering).
        // No pre/post free readings needed for out[0] — we query cuDSS's own
        // deterministic estimate instead of the pool-quantized delta.
        CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS,
                                 config, data, matA, matX, matB));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Query cuDSS's own memory estimate (available after PHASE_ANALYSIS).
        // cuDSS 0.7.1: CUDSS_DATA_MEMORY_ESTIMATES is an int64_t array whose
        // exact layout (device-permanent, device-peak, host-permanent, host-peak
        // or similar) is not documented in the installed header.  We use a
        // generous 16-element buffer and take the max of all written entries as
        // out[0]; this is a safe over-estimate (over-estimating M_block only
        // makes chunks more conservative, not OOM).  The stderr diagnostic below
        // lets us observe the real layout and refine if needed.
        int64_t mem_est[16] = {0};
        size_t written = 0;
        CUDSS_CHECK(cudssDataGet(handle, data, CUDSS_DATA_MEMORY_ESTIMATES,
                                 mem_est, sizeof(mem_est), &written));
        int n_written = (int)(written / sizeof(int64_t));
        fprintf(stderr,
                "CudssProbeDeviceBytes: written=%zu vals=[%lld,%lld,%lld,%lld,%lld,%lld]\n",
                written,
                (long long)mem_est[0], (long long)mem_est[1], (long long)mem_est[2],
                (long long)mem_est[3], (long long)mem_est[4], (long long)mem_est[5]);
        int64_t best = 0;
        for (int i = 0; i < n_written && i < 16; ++i)
            if (mem_est[i] > best) best = mem_est[i];
        out[0] = best;  // 0 if query returned nothing — caller treats as unusable

        // out[1]: absolute free device bytes (single reading, order-independent).
        size_t freeB = 0, totB = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeB, &totB));
        out[1] = (int64_t)freeB;   // budget available to cuDSS
    } catch (const std::exception& e) {
        fprintf(stderr, "CudssProbeDeviceBytesImpl: %s\n", e.what());
        rc = -1;
    }
    if (data)   { cudssHandle_t h = get_cudss_handle(); cudssDataDestroy(h, data); }
    if (matB)   cudssMatrixDestroy(matB);
    if (matX)   cudssMatrixDestroy(matX);
    if (matA)   cudssMatrixDestroy(matA);
    if (config) cudssConfigDestroy(config);
    if (d_sol)    cudaFree(d_sol);
    if (d_rhs)    cudaFree(d_rhs);
    if (d_vals)   cudaFree(d_vals);
    if (d_colIdx) cudaFree(d_colIdx);
    if (d_rowPtr) cudaFree(d_rowPtr);
    return rc;
}

namespace turbompc {
namespace cudss_sparse_kkt {

template <typename T>
void solve_sparse_kkt_impl(
    cudaStream_t stream,
    const int32_t* rowPtr,
    const int32_t* colIdx,
    const T* values,
    const T* rhs,
    T* solution,
    int32_t n,
    int32_t nnz)
{
    cudssHandle_t handle = get_cudss_handle();
    CUDSS_CHECK(cudssSetStream(handle, stream));

    const bool is_double = std::is_same_v<T, double>;
    const cudssDataType_t valueType = is_double ? CUDSS_R_64F : CUDSS_R_32F;

    // Create config (cheap — just a settings struct)
    cudssConfig_t config;
    CUDSS_CHECK(cudssConfigCreate(&config));
    cudssReorderingAlg_t reorder_alg = CUDSS_REORDERING_ALG_DEFAULT;
    CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG,
                               &reorder_alg, sizeof(reorder_alg)));

    // Create matrix descriptors (cheap — just pointer + metadata, no GPU work)
    cudssMatrix_t matA;
    CUDSS_CHECK(cudssMatrixCreateCsr(
        &matA, n, n, nnz,
        (void*)rowPtr, nullptr, (void*)colIdx, (void*)values,
        CUDSS_R_32I, CUDSS_R_32I, valueType,
        CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));

    cudssMatrix_t matX, matB;
    CUDSS_CHECK(cudssMatrixCreateDn(
        &matX, n, 1, n, (void*)solution, valueType, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(
        &matB, n, 1, n, (void*)rhs,      valueType, CUDSS_LAYOUT_COL_MAJOR));

    // Look up (or create) the cached analysis data object for this (n, nnz)
    AnalysisCacheKey cache_key{n, nnz, is_double};
    cudssData_t data = nullptr;
    bool need_analysis = false;

    {
        std::lock_guard<std::mutex> lock(g_analysis_mutex);
        auto it = g_analysis_cache.find(cache_key);
        if (it == g_analysis_cache.end()) {
            CUDSS_CHECK(cudssDataCreate(handle, &data));
            g_analysis_cache[cache_key] = data;
            need_analysis = true;
        } else {
            data = it->second;
        }
    }

    if (need_analysis) {
        // First call for this (n, nnz): run full analysis (symbolic factorization).
        // This is the expensive ~5ms step — paid only once per problem shape.
        CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS,
                                 config, data, matA, matX, matB));
    }

    // Numeric factorization + solve (always runs, uses current values)
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION,
                             config, data, matA, matX, matB));
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE,
                             config, data, matA, matX, matB));

    // Cleanup descriptors (cheap — no GPU work, data stays cached)
    cudssMatrixDestroy(matB);
    cudssMatrixDestroy(matX);
    cudssMatrixDestroy(matA);
    cudssConfigDestroy(config);
    // data and handle are cached — do not destroy
}

// Explicit instantiations
void solve_sparse_kkt_f32(
    cudaStream_t stream,
    const int32_t* rowPtr,
    const int32_t* colIdx,
    const float* values,
    const float* rhs,
    float* solution,
    int32_t n,
    int32_t nnz)
{
    solve_sparse_kkt_impl<float>(stream, rowPtr, colIdx, values, rhs, solution, n, nnz);
}

void solve_sparse_kkt_f64(
    cudaStream_t stream,
    const int32_t* rowPtr,
    const int32_t* colIdx,
    const double* values,
    const double* rhs,
    double* solution,
    int32_t n,
    int32_t nnz)
{
    solve_sparse_kkt_impl<double>(stream, rowPtr, colIdx, values, rhs, solution, n, nnz);
}

// ---------------------------------------------------------------------------
// Dense-input variant: converts dense n×n (row-major) matrix to CSR on GPU
// via cuSPARSE before calling cuDSS, avoiding any host round-trips.
// ---------------------------------------------------------------------------
template <typename T>
void solve_sparse_kkt_from_dense_impl(
    cudaStream_t stream,
    const T* dense_matrix,   // n×n row-major, device pointer
    const T* rhs,            // n-element, device pointer
    T* solution,             // n-element output, device pointer
    int32_t n)
{
    const cudaDataType value_type = std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F;

    // --- cuSPARSE dense → CSR ---
    cusparseHandle_t sp_handle;
    CUSPARSE_CHECK(cusparseCreate(&sp_handle));
    CUSPARSE_CHECK(cusparseSetStream(sp_handle, stream));

    // Dense matrix descriptor (row-major, leading dim = n)
    cusparseDnMatDescr_t mat_dense;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &mat_dense, n, n, n, (void*)dense_matrix,
        value_type, CUSPARSE_ORDER_ROW));

    // Allocate rowPtr on device
    int32_t* d_rowPtr;
    CUDA_CHECK(cudaMalloc(&d_rowPtr, (n + 1) * sizeof(int32_t)));

    // Create sparse CSR descriptor (nnz=0; rowPtr filled by analysis)
    cusparseSpMatDescr_t mat_sparse;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_sparse, n, n, 0,
        d_rowPtr, nullptr, nullptr,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, value_type));

    // Query buffer size
    size_t buf_size = 0;
    CUSPARSE_CHECK(cusparseDenseToSparse_bufferSize(
        sp_handle, mat_dense, mat_sparse,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &buf_size));

    void* d_buf = nullptr;
    if (buf_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_buf, buf_size));
    }

    // Analysis: fills rowPtr on device and updates nnz in mat_sparse
    CUSPARSE_CHECK(cusparseDenseToSparse_analysis(
        sp_handle, mat_dense, mat_sparse,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, d_buf));

    // Sync so the host can read nnz from the descriptor
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int64_t rows_tmp, cols_tmp, nnz64;
    CUSPARSE_CHECK(cusparseSpMatGetSize(mat_sparse, &rows_tmp, &cols_tmp, &nnz64));
    const int32_t nnz = static_cast<int32_t>(nnz64);

    // Allocate colIdx and values (exact size, no waste)
    int32_t* d_colIdx;
    T*       d_values;
    CUDA_CHECK(cudaMalloc(&d_colIdx, nnz * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(T)));

    CUSPARSE_CHECK(cusparseCsrSetPointers(mat_sparse, d_rowPtr, d_colIdx, d_values));

    // Conversion: fill colIdx and values on device
    CUSPARSE_CHECK(cusparseDenseToSparse_convert(
        sp_handle, mat_dense, mat_sparse,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, d_buf));

    // --- cuDSS solve using the CSR produced above ---
    solve_sparse_kkt_impl<T>(stream, d_rowPtr, d_colIdx, d_values, rhs, solution, n, nnz);

    // Cleanup
    cudaFree(d_values);
    cudaFree(d_colIdx);
    cudaFree(d_rowPtr);
    if (d_buf) cudaFree(d_buf);
    CUSPARSE_CHECK(cusparseDestroyDnMat(mat_dense));
    CUSPARSE_CHECK(cusparseDestroySpMat(mat_sparse));
    CUSPARSE_CHECK(cusparseDestroy(sp_handle));
}

void solve_sparse_kkt_from_dense_f32(
    cudaStream_t stream,
    const float* dense_matrix,
    const float* rhs,
    float* solution,
    int32_t n)
{
    solve_sparse_kkt_from_dense_impl<float>(stream, dense_matrix, rhs, solution, n);
}

void solve_sparse_kkt_from_dense_f64(
    cudaStream_t stream,
    const double* dense_matrix,
    const double* rhs,
    double* solution,
    int32_t n)
{
    solve_sparse_kkt_from_dense_impl<double>(stream, dense_matrix, rhs, solution, n);
}

}  // namespace cudss_sparse_kkt
}  // namespace turbompc
