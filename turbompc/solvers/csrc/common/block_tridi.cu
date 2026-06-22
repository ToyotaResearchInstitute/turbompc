#include "block_tridi.cuh"

template<typename T>
__device__ void blkTridiMatvecRow(
    T* y_out,
    const T* S_row,
    const T* x_prev,
    const T* x_curr,
    const T* x_next,
    int n)
{
    // S_row is (n, 3n): columns [0..n) = sub, [n..2n) = diag, [2n..3n) = super
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T sum = 0;
        const T* row = S_row + i * 3 * n;
        if (x_prev) {
            for (int j = 0; j < n; ++j) sum += row[j] * x_prev[j];
        }
        for (int j = 0; j < n; ++j) sum += row[n + j] * x_curr[j];
        if (x_next) {
            for (int j = 0; j < n; ++j) sum += row[2*n + j] * x_next[j];
        }
        y_out[i] = sum;
    }
}

template __device__ void blkTridiMatvecRow<float>(
    float*, const float*, const float*, const float*, const float*, int);
template __device__ void blkTridiMatvecRow<double>(
    double*, const double*, const double*, const double*, const double*, int);

// ---------- CSR assembly kernel ----------
template<typename T>
__global__ void blkTridiToCSRKernel(
    const T* __restrict__ S,
    int32_t* __restrict__ indptr,
    int32_t* __restrict__ indices,
    T* __restrict__ data,
    int T_blocks,
    int n)
{
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = T_blocks * n;
    if (global_row >= total_rows) return;

    int t = global_row / n;
    int i = global_row % n;

    int num_blocks_in_row = 1 + (t > 0 ? 1 : 0) + (t < T_blocks - 1 ? 1 : 0);
    int nnz_in_row = num_blocks_in_row * n;

    int offset;
    if (T_blocks == 1) {
        offset = global_row * n;
    } else if (t == 0) {
        offset = global_row * 2 * n;
    } else if (t == T_blocks - 1) {
        offset = n * 2 * n + (T_blocks - 2) * n * 3 * n + i * 2 * n;
    } else {
        offset = n * 2 * n + (t - 1) * n * 3 * n + i * 3 * n;
    }

    indptr[global_row] = offset;
    if (global_row == total_rows - 1) {
        indptr[total_rows] = offset + nnz_in_row;
    }

    const T* S_row = S + t * n * 3 * n + i * 3 * n;
    int write_pos = offset;

    if (t > 0) {
        for (int j = 0; j < n; ++j) {
            indices[write_pos] = (t - 1) * n + j;
            data[write_pos] = S_row[j];
            write_pos++;
        }
    }
    for (int j = 0; j < n; ++j) {
        indices[write_pos] = t * n + j;
        data[write_pos] = S_row[n + j];
        write_pos++;
    }
    if (t < T_blocks - 1) {
        for (int j = 0; j < n; ++j) {
            indices[write_pos] = (t + 1) * n + j;
            data[write_pos] = S_row[2 * n + j];
            write_pos++;
        }
    }
}

void blkTridiToCSR(
    const float* S_dev, int32_t* indptr, int32_t* indices, float* data,
    int T_blocks, int n, cudaStream_t stream)
{
    int total_rows = T_blocks * n;
    int threads = 256;
    int blocks = (total_rows + threads - 1) / threads;
    blkTridiToCSRKernel<float><<<blocks, threads, 0, stream>>>(
        S_dev, indptr, indices, data, T_blocks, n);
}

void blkTridiToCSR_f64(
    const double* S_dev, int32_t* indptr, int32_t* indices, double* data,
    int T_blocks, int n, cudaStream_t stream)
{
    int total_rows = T_blocks * n;
    int threads = 256;
    int blocks = (total_rows + threads - 1) / threads;
    blkTridiToCSRKernel<double><<<blocks, threads, 0, stream>>>(
        S_dev, indptr, indices, data, T_blocks, n);
}

// ---------- Values-only CSR kernel ----------
// Updates only the data array; indptr and indices are unchanged.
template<typename T>
__global__ void blkTridiToCSR_DataOnlyKernel(
    const T* __restrict__ S,
    T* __restrict__ data,
    int T_blocks,
    int n)
{
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = T_blocks * n;
    if (global_row >= total_rows) return;

    int t = global_row / n;
    int i = global_row % n;

    // Compute the write offset (same logic as the full kernel)
    int offset;
    if (T_blocks == 1) {
        offset = global_row * n;
    } else if (t == 0) {
        offset = global_row * 2 * n;
    } else if (t == T_blocks - 1) {
        offset = n * 2 * n + (T_blocks - 2) * n * 3 * n + i * 2 * n;
    } else {
        offset = n * 2 * n + (t - 1) * n * 3 * n + i * 3 * n;
    }

    const T* S_row = S + t * n * 3 * n + i * 3 * n;
    int write_pos = offset;

    if (t > 0) {
        for (int j = 0; j < n; ++j) {
            data[write_pos] = S_row[j];
            write_pos++;
        }
    }
    for (int j = 0; j < n; ++j) {
        data[write_pos] = S_row[n + j];
        write_pos++;
    }
    if (t < T_blocks - 1) {
        for (int j = 0; j < n; ++j) {
            data[write_pos] = S_row[2 * n + j];
            write_pos++;
        }
    }
}

void blkTridiToCSR_data_only(
    const float* S_dev, float* data,
    int T_blocks, int n, cudaStream_t stream)
{
    int total_rows = T_blocks * n;
    int threads = 256;
    int blocks = (total_rows + threads - 1) / threads;
    blkTridiToCSR_DataOnlyKernel<float><<<blocks, threads, 0, stream>>>(
        S_dev, data, T_blocks, n);
}

void blkTridiToCSR_data_only_f64(
    const double* S_dev, double* data,
    int T_blocks, int n, cudaStream_t stream)
{
    int total_rows = T_blocks * n;
    int threads = 256;
    int blocks = (total_rows + threads - 1) / threads;
    blkTridiToCSR_DataOnlyKernel<double><<<blocks, threads, 0, stream>>>(
        S_dev, data, T_blocks, n);
}

// ========== Batched block-diagonal CSR assembly ==========
// Assembles Nb independent block-tridiagonal systems into ONE block-diagonal
// CSR matrix of size (Nb*T*n) x (Nb*T*n).
// S_dev: [Nb, T, n, 3n] contiguous (each batch element is T*n*3n floats apart)

__host__ __device__ static size_t computeNnzPerSystem(int T, int n) {
    if (T == 1) return (size_t)n * n;
    return (size_t)(3 * T - 2) * n * n;
}

// Compute the NNZ offset for row (global_row within one system) in one system's CSR.
__host__ __device__ static int singleSystemRowOffset(int global_row, int T_blocks, int n) {
    int t = global_row / n;
    int i = global_row % n;
    int offset;
    if (T_blocks == 1) {
        offset = global_row * n;
    } else if (t == 0) {
        offset = global_row * 2 * n;
    } else if (t == T_blocks - 1) {
        offset = n * 2 * n + (T_blocks - 2) * n * 3 * n + i * 2 * n;
    } else {
        offset = n * 2 * n + (t - 1) * n * 3 * n + i * 3 * n;
    }
    return offset;
}

template<typename T>
__global__ void batchedBlkTridiToCSRKernel(
    const T* __restrict__ S,         // [Nb, T, n, 3n]
    int32_t* __restrict__ indptr,    // [Nb*T*n + 1]
    int32_t* __restrict__ indices,   // [Nb * nnz_per_system]
    T* __restrict__ data,            // [Nb * nnz_per_system]
    int Nb,
    int T_blocks,
    int n)
{
    int total_rows_all = Nb * T_blocks * n;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_rows_all) return;

    int rows_per_sys = T_blocks * n;
    int b = gid / rows_per_sys;        // batch index
    int local_row = gid % rows_per_sys; // row within this system
    int t = local_row / n;
    int i = local_row % n;

    size_t nnz_per_sys = computeNnzPerSystem(T_blocks, n);
    // Global offset into data/indices for this batch element
    size_t batch_nnz_offset = (size_t)b * nnz_per_sys;
    // Row offset within this system
    int local_offset = singleSystemRowOffset(local_row, T_blocks, n);
    // Global offset = batch_nnz_offset + local_offset
    size_t global_offset = batch_nnz_offset + local_offset;

    indptr[gid] = (int32_t)global_offset;
    if (gid == total_rows_all - 1) {
        indptr[total_rows_all] = (int32_t)(batch_nnz_offset + nnz_per_sys);
    }

    // Column indices are shifted by b * rows_per_sys for block-diagonal structure
    int col_base = b * rows_per_sys;

    const T* S_row = S + (size_t)b * T_blocks * n * 3 * n + (size_t)t * n * 3 * n + (size_t)i * 3 * n;
    size_t wp = global_offset;

    if (t > 0) {
        for (int j = 0; j < n; ++j) {
            indices[wp] = col_base + (t - 1) * n + j;
            data[wp] = S_row[j];
            wp++;
        }
    }
    for (int j = 0; j < n; ++j) {
        indices[wp] = col_base + t * n + j;
        data[wp] = S_row[n + j];
        wp++;
    }
    if (t < T_blocks - 1) {
        for (int j = 0; j < n; ++j) {
            indices[wp] = col_base + (t + 1) * n + j;
            data[wp] = S_row[2 * n + j];
            wp++;
        }
    }
}

void batchedBlkTridiToCSR(
    const float* S_dev, int32_t* indptr, int32_t* indices, float* data,
    int Nb, int T_blocks, int n, cudaStream_t stream)
{
    int total_rows = Nb * T_blocks * n;
    int threads = 256;
    int blocks = (total_rows + threads - 1) / threads;
    batchedBlkTridiToCSRKernel<float><<<blocks, threads, 0, stream>>>(
        S_dev, indptr, indices, data, Nb, T_blocks, n);
}

void batchedBlkTridiToCSR_f64(
    const double* S_dev, int32_t* indptr, int32_t* indices, double* data,
    int Nb, int T_blocks, int n, cudaStream_t stream)
{
    int total_rows = Nb * T_blocks * n;
    int threads = 256;
    int blocks = (total_rows + threads - 1) / threads;
    batchedBlkTridiToCSRKernel<double><<<blocks, threads, 0, stream>>>(
        S_dev, indptr, indices, data, Nb, T_blocks, n);
}

// Batched values-only variant
template<typename T>
__global__ void batchedBlkTridiToCSR_DataOnlyKernel(
    const T* __restrict__ S,
    T* __restrict__ data,
    int Nb,
    int T_blocks,
    int n)
{
    int total_rows_all = Nb * T_blocks * n;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_rows_all) return;

    int rows_per_sys = T_blocks * n;
    int b = gid / rows_per_sys;
    int local_row = gid % rows_per_sys;
    int t = local_row / n;
    int i = local_row % n;

    size_t nnz_per_sys = computeNnzPerSystem(T_blocks, n);
    size_t batch_nnz_offset = (size_t)b * nnz_per_sys;
    int local_offset = singleSystemRowOffset(local_row, T_blocks, n);
    size_t wp = batch_nnz_offset + local_offset;

    const T* S_row = S + (size_t)b * T_blocks * n * 3 * n + (size_t)t * n * 3 * n + (size_t)i * 3 * n;

    if (t > 0) {
        for (int j = 0; j < n; ++j) {
            data[wp] = S_row[j];
            wp++;
        }
    }
    for (int j = 0; j < n; ++j) {
        data[wp] = S_row[n + j];
        wp++;
    }
    if (t < T_blocks - 1) {
        for (int j = 0; j < n; ++j) {
            data[wp] = S_row[2 * n + j];
            wp++;
        }
    }
}

void batchedBlkTridiToCSR_data_only(
    const float* S_dev, float* data,
    int Nb, int T_blocks, int n, cudaStream_t stream)
{
    int total_rows = Nb * T_blocks * n;
    int threads = 256;
    int blocks = (total_rows + threads - 1) / threads;
    batchedBlkTridiToCSR_DataOnlyKernel<float><<<blocks, threads, 0, stream>>>(
        S_dev, data, Nb, T_blocks, n);
}

void batchedBlkTridiToCSR_data_only_f64(
    const double* S_dev, double* data,
    int Nb, int T_blocks, int n, cudaStream_t stream)
{
    int total_rows = Nb * T_blocks * n;
    int threads = 256;
    int blocks = (total_rows + threads - 1) / threads;
    batchedBlkTridiToCSR_DataOnlyKernel<double><<<blocks, threads, 0, stream>>>(
        S_dev, data, Nb, T_blocks, n);
}
