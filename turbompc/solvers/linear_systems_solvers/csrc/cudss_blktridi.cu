#include "cudss_blktridi.cuh"
#include "block_tridi.cuh"
#include <cudss.h>
#include <mutex>
#include <unordered_map>

struct CudssPlanEntry {
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t matA = nullptr;
    cudssMatrix_t matX = nullptr;
    cudssMatrix_t matB = nullptr;
    int32_t* indptr = nullptr;
    int32_t* indices = nullptr;
    void* csr_data = nullptr;
    void* x_buf = nullptr;
    int32_t T_blocks = 0;
    int32_t n = 0;
    size_t nnz = 0;
    bool planned = false;
    bool structure_written = false;  // indptr/indices populated
    std::mutex entry_mutex;          // per-entry lock for solve phase

    CudssPlanEntry() = default;

    // Non-copyable (has mutex), but movable for map insertion
    CudssPlanEntry(CudssPlanEntry&& o) noexcept
        : handle(o.handle), config(o.config), data(o.data),
          matA(o.matA), matX(o.matX), matB(o.matB),
          indptr(o.indptr), indices(o.indices), csr_data(o.csr_data),
          x_buf(o.x_buf), T_blocks(o.T_blocks), n(o.n), nnz(o.nnz),
          planned(o.planned), structure_written(o.structure_written)
    {
        o.handle = nullptr; o.config = nullptr; o.data = nullptr;
        o.matA = nullptr; o.matX = nullptr; o.matB = nullptr;
        o.indptr = nullptr; o.indices = nullptr; o.csr_data = nullptr;
        o.x_buf = nullptr;
    }

    CudssPlanEntry(const CudssPlanEntry&) = delete;
    CudssPlanEntry& operator=(const CudssPlanEntry&) = delete;
    CudssPlanEntry& operator=(CudssPlanEntry&&) = delete;
};

static std::mutex g_mutex;  // protects cache lookup/insert only
static std::unordered_map<uint64_t, CudssPlanEntry> g_cache;

static uint64_t cacheKey(int32_t Nb, int32_t T, int32_t n, int dtype_bytes) {
    // Pack (Nb, T, n, dtype_bytes) into 64 bits
    return ((uint64_t)Nb << 40) | ((uint64_t)T << 20) | ((uint64_t)n << 8) | (uint64_t)dtype_bytes;
}

static size_t computeNnz(int T, int n) {
    if (T == 1) return (size_t)n * n;
    return (size_t)(3 * T - 2) * n * n;
}

template<typename T>
static void cudssBlkTridiSolveImpl(
    cudaStream_t stream,
    T* x_out_dev,
    const T* S_dev,
    const T* rhs_dev,
    int32_t T_blocks,
    int32_t n,
    int32_t Nb = 1)
{
    int total = Nb * T_blocks * n;
    size_t nnz_per_sys = computeNnz(T_blocks, n);
    size_t nnz = (size_t)Nb * nnz_per_sys;
    uint64_t key = cacheKey(Nb, T_blocks, n, sizeof(T));

    // Phase 1: cache lookup/insert under global lock (fast)
    CudssPlanEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_cache.find(key);
        if (it == g_cache.end()) {
            CudssPlanEntry e{};
            e.T_blocks = T_blocks;
            e.n = n;
            e.nnz = nnz;
            e.planned = false;
            e.structure_written = false;

            cudaMalloc(&e.indptr, (total + 1) * sizeof(int32_t));
            cudaMalloc(&e.indices, nnz * sizeof(int32_t));
            cudaMalloc(&e.csr_data, nnz * sizeof(T));
            cudaMalloc(&e.x_buf, total * sizeof(T));

            cudssCreate(&e.handle);
            cudssConfigCreate(&e.config);
            cudssDataCreate(e.handle, &e.data);

            auto [ins, _] = g_cache.emplace(key, std::move(e));
            entry = &ins->second;
        } else {
            entry = &it->second;
        }
    }
    // Global lock released — only hold per-entry lock for the solve

    std::lock_guard<std::mutex> entry_lock(entry->entry_mutex);

    cudssSetStream(entry->handle, stream);

    // CSR assembly: full on first call, values-only on subsequent calls
    if (!entry->structure_written) {
        if (Nb == 1) {
            if constexpr (std::is_same_v<T, float>) {
                blkTridiToCSR(S_dev, entry->indptr, entry->indices,
                              (float*)entry->csr_data, T_blocks, n, stream);
            } else {
                blkTridiToCSR_f64(S_dev, entry->indptr, entry->indices,
                                  (double*)entry->csr_data, T_blocks, n, stream);
            }
        } else {
            if constexpr (std::is_same_v<T, float>) {
                batchedBlkTridiToCSR(S_dev, entry->indptr, entry->indices,
                                     (float*)entry->csr_data, Nb, T_blocks, n, stream);
            } else {
                batchedBlkTridiToCSR_f64(S_dev, entry->indptr, entry->indices,
                                         (double*)entry->csr_data, Nb, T_blocks, n, stream);
            }
        }
        entry->structure_written = true;
    } else {
        // Only update values — indptr/indices are unchanged
        if (Nb == 1) {
            if constexpr (std::is_same_v<T, float>) {
                blkTridiToCSR_data_only(S_dev, (float*)entry->csr_data,
                                        T_blocks, n, stream);
            } else {
                blkTridiToCSR_data_only_f64(S_dev, (double*)entry->csr_data,
                                            T_blocks, n, stream);
            }
        } else {
            if constexpr (std::is_same_v<T, float>) {
                batchedBlkTridiToCSR_data_only(S_dev, (float*)entry->csr_data,
                                               Nb, T_blocks, n, stream);
            } else {
                batchedBlkTridiToCSR_data_only_f64(S_dev, (double*)entry->csr_data,
                                                    Nb, T_blocks, n, stream);
            }
        }
    }

    cudssDataType_t valueType = std::is_same_v<T, float> ? CUDSS_R_32F : CUDSS_R_64F;

    // For batched: rhs is [Nb, T, n] contiguous, treated as a single dense vector of length total.
    // Create cuDSS matrix descriptors on first use; update pointers on subsequent calls.
    if (entry->matA == nullptr) {
        cudssMatrixCreateCsr(&entry->matA, total, total, nnz,
                             entry->indptr, nullptr, entry->indices, entry->csr_data,
                             CUDSS_R_32I, CUDSS_R_32I, valueType,
                             CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO);
        cudssMatrixCreateDn(&entry->matX, total, 1, total, entry->x_buf,
                            valueType, CUDSS_LAYOUT_COL_MAJOR);
        cudssMatrixCreateDn(&entry->matB, total, 1, total, (void*)rhs_dev,
                            valueType, CUDSS_LAYOUT_COL_MAJOR);
    } else {
        // Update the rhs data pointer (changes each call); CSR data pointer is stable.
        cudssMatrixSetValues(entry->matB, (void*)rhs_dev);
    }

    cudssReorderingAlg_t reorder_alg = CUDSS_REORDERING_ALG_DEFAULT;
    cudssConfigSet(entry->config, CUDSS_CONFIG_REORDERING_ALG, &reorder_alg, sizeof(reorder_alg));

    if (!entry->planned) {
        cudssExecute(entry->handle, CUDSS_PHASE_ANALYSIS, entry->config,
                     entry->data, entry->matA, entry->matX, entry->matB);
        entry->planned = true;
    }

    cudssExecute(entry->handle, CUDSS_PHASE_FACTORIZATION, entry->config,
                 entry->data, entry->matA, entry->matX, entry->matB);

    cudssExecute(entry->handle, CUDSS_PHASE_SOLVE, entry->config,
                 entry->data, entry->matA, entry->matX, entry->matB);

    cudaMemcpyAsync(x_out_dev, entry->x_buf, total * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
}

void CudssBlkTridiSolveF32(
    cudaStream_t stream, float* x_out, const float* S, const float* rhs,
    int32_t T_blocks, int32_t n, int32_t Nb)
{
    cudssBlkTridiSolveImpl<float>(stream, x_out, S, rhs, T_blocks, n, Nb);
}

void CudssBlkTridiSolveF64(
    cudaStream_t stream, double* x_out, const double* S, const double* rhs,
    int32_t T_blocks, int32_t n, int32_t Nb)
{
    cudssBlkTridiSolveImpl<double>(stream, x_out, S, rhs, T_blocks, n, Nb);
}
