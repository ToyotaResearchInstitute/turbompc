#pragma once

#include <cstdint>

// Inequality treatment selected by the solver, shared by every ADMM backend.
// Keep in sync with turbompc.solvers.constraint_handling.ConstraintHandling.
enum class ConstraintHandling : int32_t {
    Projection = 0,
    LogBarrier = 1,
};
