#pragma once

#include <cmath>
#include <cuda_runtime.h>

template<typename T>
__device__ T ADMMBlockMax(T value, T* warp_buffer)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, value, offset);
        value = other > value ? other : value;
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warp_buffer[warp] = value;
    __syncthreads();
    T result = T(0);
    if (warp == 0) {
        int num_warps = (blockDim.x + 31) >> 5;
        result = lane < num_warps ? warp_buffer[lane] : T(0);
        for (int offset = 16; offset > 0; offset /= 2) {
            T other = __shfl_down_sync(0xffffffff, result, offset);
            result = other > result ? other : result;
        }
    }
    __syncthreads();
    return result;
}

template<typename T>
__device__ T ADMMBlockSum(T value, T* warp_buffer)
{
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warp_buffer[warp] = value;
    __syncthreads();
    T result = T(0);
    if (warp == 0) {
        int num_warps = (blockDim.x + 31) >> 5;
        result = lane < num_warps ? warp_buffer[lane] : T(0);
        for (int offset = 16; offset > 0; offset /= 2)
            result += __shfl_down_sync(0xffffffff, result, offset);
    }
    __syncthreads();
    return result;
}

// Metrics: dy norm, primal stationarity, primal support, dx norm,
// Hessian direction, objective direction, bound/equality direction violation.
template<typename T>
__device__ void ComputeADMMInfeasibilityMetrics(
    T* metrics,
    const T* x, const T* yg, const T* yf0, const T* yfdyn, const T* xi,
    const T* previous_x, const T* previous_yg,
    const T* previous_yf0, const T* previous_yfdyn, const T* previous_xi,
    const T* D, const T* E, const T* q,
    const T* A0, const T* Am, const T* Ap, const T* G,
    const T* lower, const T* upper, const T* c0, const T* cdyn,
    T slack_weight, bool use_slack,
    int Tn, int n, int nx, int n0, int m,
    T* warp_buffer)
{
    int tid = threadIdx.x;
    int N = Tn - 1;
    int tn = Tn * n;
    int tm = Tn * m;
    int ndyn = N * nx;

    T dy_norm_local = T(0);
    for (int i = tid; i < n0; i += blockDim.x)
        dy_norm_local = fmax(dy_norm_local, fabs(yf0[i] - previous_yf0[i]));
    for (int i = tid; i < ndyn; i += blockDim.x)
        dy_norm_local = fmax(dy_norm_local, fabs(yfdyn[i] - previous_yfdyn[i]));
    for (int i = tid; i < tm; i += blockDim.x)
        dy_norm_local = fmax(dy_norm_local, fabs(yg[i] - previous_yg[i]));
    T primal_stationarity_local = T(0);
    for (int index = tid; index < tn; index += blockDim.x) {
        int t = index / n;
        int j = index % n;
        T value = T(0);
        if (t == 0)
            for (int r = 0; r < n0; ++r)
                value += A0[r * n + j] * (yf0[r] - previous_yf0[r]);
        if (t > 0)
            for (int r = 0; r < nx; ++r) {
                int row = (t - 1) * nx + r;
                value += Ap[row * n + j] * (yfdyn[row] - previous_yfdyn[row]);
            }
        if (t < N)
            for (int r = 0; r < nx; ++r) {
                int row = t * nx + r;
                value += Am[row * n + j] * (yfdyn[row] - previous_yfdyn[row]);
            }
        for (int r = 0; r < m; ++r) {
            int row = t * m + r;
            value += G[row * n + j] * (yg[row] - previous_yg[row]);
        }
        primal_stationarity_local = fmax(primal_stationarity_local, fabs(value));
    }
    if (use_slack)
        for (int i = tid; i < tm; i += blockDim.x)
            primal_stationarity_local = fmax(
                primal_stationarity_local, fabs(yg[i] - previous_yg[i]));
    T support_local = T(0);
    for (int i = tid; i < n0; i += blockDim.x)
        support_local += c0[i] * (yf0[i] - previous_yf0[i]);
    for (int i = tid; i < ndyn; i += blockDim.x)
        support_local += cdyn[i] * (yfdyn[i] - previous_yfdyn[i]);
    for (int i = tid; i < tm; i += blockDim.x) {
        T direction = yg[i] - previous_yg[i];
        if (direction > T(0))
            support_local += upper[i] == T(INFINITY)
                ? T(INFINITY) : upper[i] * direction;
        else if (direction < T(0))
            support_local += lower[i] == T(-INFINITY)
                ? T(INFINITY) : lower[i] * direction;
    }
    T dx_norm_local = T(0);
    for (int i = tid; i < tn; i += blockDim.x)
        dx_norm_local = fmax(dx_norm_local, fabs(x[i] - previous_x[i]));
    if (use_slack)
        for (int i = tid; i < tm; i += blockDim.x)
            dx_norm_local = fmax(dx_norm_local, fabs(xi[i] - previous_xi[i]));
    T hessian_local = T(0);
    T objective_local = T(0);
    for (int index = tid; index < tn; index += blockDim.x) {
        int t = index / n;
        int j = index % n;
        T value = T(0);
        for (int k = 0; k < n; ++k)
            value += D[index * n + k] * (x[t * n + k] - previous_x[t * n + k]);
        if (t > 0)
            for (int k = 0; k < n; ++k)
                value += E[((t - 1) * n + j) * n + k]
                    * (x[(t - 1) * n + k] - previous_x[(t - 1) * n + k]);
        if (t < N)
            for (int k = 0; k < n; ++k)
                value += E[(t * n + k) * n + j]
                    * (x[(t + 1) * n + k] - previous_x[(t + 1) * n + k]);
        hessian_local = fmax(hessian_local, fabs(value));
        objective_local += q[index] * (x[index] - previous_x[index]);
    }
    if (use_slack)
        for (int i = tid; i < tm; i += blockDim.x)
            hessian_local = fmax(
                hessian_local, fabs(slack_weight * (xi[i] - previous_xi[i])));
    T direction_violation_local = T(0);
    for (int row = tid; row < n0; row += blockDim.x) {
        T value = T(0);
        for (int j = 0; j < n; ++j)
            value += A0[row * n + j] * (x[j] - previous_x[j]);
        direction_violation_local = fmax(direction_violation_local, fabs(value));
    }
    for (int row = tid; row < ndyn; row += blockDim.x) {
        int t = row / nx;
        T value = T(0);
        for (int j = 0; j < n; ++j)
            value += Am[row * n + j] * (x[t * n + j] - previous_x[t * n + j])
                + Ap[row * n + j]
                    * (x[(t + 1) * n + j] - previous_x[(t + 1) * n + j]);
        direction_violation_local = fmax(direction_violation_local, fabs(value));
    }
    for (int row = tid; row < tm; row += blockDim.x) {
        int t = row / m;
        T value = T(0);
        for (int j = 0; j < n; ++j)
            value += G[row * n + j] * (x[t * n + j] - previous_x[t * n + j]);
        if (use_slack) value += xi[row] - previous_xi[row];
        T violation = T(0);
        bool lower_finite = isfinite(lower[row]);
        bool upper_finite = isfinite(upper[row]);
        if (lower_finite && upper_finite) violation = fabs(value);
        else if (lower_finite) violation = fmax(-value, T(0));
        else if (upper_finite) violation = fmax(value, T(0));
        direction_violation_local = fmax(direction_violation_local, violation);
    }
    // Reduce all seven certificate metrics in one coordinated block reduction.
    for (int offset = 16; offset > 0; offset /= 2) {
        dy_norm_local = fmax(
            dy_norm_local,
            __shfl_down_sync(0xffffffff, dy_norm_local, offset));
        primal_stationarity_local = fmax(
            primal_stationarity_local,
            __shfl_down_sync(0xffffffff, primal_stationarity_local, offset));
        support_local += __shfl_down_sync(0xffffffff, support_local, offset);
        dx_norm_local = fmax(
            dx_norm_local,
            __shfl_down_sync(0xffffffff, dx_norm_local, offset));
        hessian_local = fmax(
            hessian_local,
            __shfl_down_sync(0xffffffff, hessian_local, offset));
        objective_local += __shfl_down_sync(0xffffffff, objective_local, offset);
        direction_violation_local = fmax(
            direction_violation_local,
            __shfl_down_sync(0xffffffff, direction_violation_local, offset));
    }
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = (blockDim.x + 31) >> 5;
    if (lane == 0) {
        T* warp_metrics = warp_buffer + 7 * warp;
        warp_metrics[0] = dy_norm_local;
        warp_metrics[1] = primal_stationarity_local;
        warp_metrics[2] = support_local;
        warp_metrics[3] = dx_norm_local;
        warp_metrics[4] = hessian_local;
        warp_metrics[5] = objective_local;
        warp_metrics[6] = direction_violation_local;
    }
    __syncthreads();
    if (warp == 0) {
        T reduced[7];
        #pragma unroll
        for (int metric = 0; metric < 7; ++metric)
            reduced[metric] = lane < num_warps
                ? warp_buffer[7 * lane + metric] : T(0);
        for (int offset = 16; offset > 0; offset /= 2) {
            reduced[0] = fmax(
                reduced[0], __shfl_down_sync(0xffffffff, reduced[0], offset));
            reduced[1] = fmax(
                reduced[1], __shfl_down_sync(0xffffffff, reduced[1], offset));
            reduced[2] += __shfl_down_sync(0xffffffff, reduced[2], offset);
            reduced[3] = fmax(
                reduced[3], __shfl_down_sync(0xffffffff, reduced[3], offset));
            reduced[4] = fmax(
                reduced[4], __shfl_down_sync(0xffffffff, reduced[4], offset));
            reduced[5] += __shfl_down_sync(0xffffffff, reduced[5], offset);
            reduced[6] = fmax(
                reduced[6], __shfl_down_sync(0xffffffff, reduced[6], offset));
        }
        if (lane == 0) {
            #pragma unroll
            for (int metric = 0; metric < 7; ++metric)
                metrics[metric] = reduced[metric];
        }
    }
    __syncthreads();
}
