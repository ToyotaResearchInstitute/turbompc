#pragma once

#include <cstdint>

// Keep in sync with turbompc.solvers.status.ADMMStatus.
constexpr int32_t ADMM_STATUS_NOT_RUN = -1;
constexpr int32_t ADMM_STATUS_SOLVED = 0;
constexpr int32_t ADMM_STATUS_MAX_ITER = 1;
constexpr int32_t ADMM_STATUS_PRIMAL_INFEASIBLE = 2;
constexpr int32_t ADMM_STATUS_DUAL_INFEASIBLE = 3;
constexpr int32_t ADMM_STATUS_NONFINITE = 4;

__host__ __device__ inline int32_t SelectADMMStatus(
    bool nonfinite,
    bool solved,
    bool primal_infeasible,
    bool dual_infeasible,
    bool iteration_limit_reached)
{
    if (nonfinite) return ADMM_STATUS_NONFINITE;
    if (solved) return ADMM_STATUS_SOLVED;
    if (primal_infeasible) return ADMM_STATUS_PRIMAL_INFEASIBLE;
    if (dual_infeasible) return ADMM_STATUS_DUAL_INFEASIBLE;
    if (iteration_limit_reached) return ADMM_STATUS_MAX_ITER;
    return ADMM_STATUS_NOT_RUN;
}
