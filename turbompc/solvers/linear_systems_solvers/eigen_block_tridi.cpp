#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <xla/ffi/api/c_api.h>
#include <xla/ffi/api/ffi.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace ffi = xla::ffi;
namespace py = pybind11;

using BufF32 = ffi::Buffer<ffi::DataType::F32>;
using BufF64 = ffi::Buffer<ffi::DataType::F64>;
using BufPred = ffi::Buffer<ffi::DataType::PRED>;

template <typename Scalar>
class BlockTridiSolver {
 public:
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using RowMatrix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using ConstRowMap = Eigen::Map<const RowMatrix>;

  BlockTridiSolver(Eigen::Index T, Eigen::Index n) : T_(T), n_(n) {
    diagonal_work_.resize(n_, n_);
    transpose_work_.resize(n_, n_);

    const Eigen::Index off_diagonal_count = T_ > 1 ? T_ - 1 : 0;
    upper_.resize(off_diagonal_count);
    multipliers_.resize(off_diagonal_count);
    for (Eigen::Index t = 0; t < off_diagonal_count; ++t) {
      upper_[t].resize(n_, n_);
      multipliers_[t].resize(n_, n_);
    }

    factors_.reserve(T_);
    for (Eigen::Index t = 0; t < T_; ++t) {
      factors_.emplace_back(n_);
    }

    reduced_rhs_.resize(T_);
    solution_.resize(T_);
    for (Eigen::Index t = 0; t < T_; ++t) {
      reduced_rhs_[t].resize(n_);
      solution_[t].resize(n_);
    }
  }

  void Factorize(const Scalar* blocks) {
    factorized_ = false;

    ConstRowMap first(blocks, n_, 3 * n_);
    diagonal_work_ = first.middleCols(n_, n_);
    Factor(0, diagonal_work_);
    if (T_ > 1) {
      upper_[0] = first.rightCols(n_);
    }

    for (Eigen::Index t = 1; t < T_; ++t) {
      ConstRowMap row(blocks + t * n_ * 3 * n_, n_, 3 * n_);
      transpose_work_ = row.leftCols(n_).transpose();
      multipliers_[t - 1] = factors_[t - 1].solve(transpose_work_).transpose();
      diagonal_work_.noalias() =
          row.middleCols(n_, n_) - multipliers_[t - 1] * upper_[t - 1];
      Factor(t, diagonal_work_);
      if (t + 1 < T_) {
        upper_[t] = row.rightCols(n_);
      }
    }

    factorized_ = true;
  }

  void Solve(const Scalar* rhs, Scalar* output) {
    if (!factorized_) {
      throw std::logic_error(
          "Eigen solve requires the workspace to be factorized");
    }
    reduced_rhs_[0] = Eigen::Map<const Vector>(rhs, n_);
    for (Eigen::Index t = 1; t < T_; ++t) {
      reduced_rhs_[t] = Eigen::Map<const Vector>(rhs + t * n_, n_) -
                        multipliers_[t - 1] * reduced_rhs_[t - 1];
    }

    solution_[T_ - 1] = factors_[T_ - 1].solve(reduced_rhs_[T_ - 1]);
    for (Eigen::Index t = T_ - 1; t > 0; --t) {
      const Eigen::Index previous = t - 1;
      solution_[previous] = factors_[previous].solve(
          reduced_rhs_[previous] - upper_[previous] * solution_[t]);
    }

    for (Eigen::Index t = 0; t < T_; ++t) {
      if (!solution_[t].allFinite()) {
        throw std::runtime_error("Eigen LDLT produced a non-finite solution");
      }
      Eigen::Map<Vector> output_block(output + t * n_, n_);
      output_block = solution_[t];
    }
  }

 private:
  void Factor(Eigen::Index t, const Matrix& diagonal) {
    factors_[t].compute(diagonal);
    if (factors_[t].info() != Eigen::Success || !factors_[t].isPositive()) {
      throw std::runtime_error(
          "Eigen LDLT failed: Schur matrix is not positive definite");
    }
  }

  bool factorized_ = false;
  Eigen::Index T_ = 0;
  Eigen::Index n_ = 0;
  Matrix diagonal_work_;
  Matrix transpose_work_;
  std::vector<Matrix> upper_;
  std::vector<Matrix> multipliers_;
  std::vector<Eigen::LDLT<Matrix>> factors_;
  std::vector<Vector> reduced_rhs_;
  std::vector<Vector> solution_;
};

enum class ScalarType { kFloat32, kFloat64 };

class WorkspaceBase {
public:
  virtual ~WorkspaceBase() = default;
  virtual ScalarType scalar_type() const noexcept = 0;
  virtual bool matches(Eigen::Index batch_size, Eigen::Index T,
                       Eigen::Index n) const noexcept = 0;
};

struct WorkspaceEntry {
  std::mutex execution_mutex;
  std::unique_ptr<WorkspaceBase> capacity;
};

template <typename Scalar>
class EigenBlockTridiWorkspace final : public WorkspaceBase {
public:
  EigenBlockTridiWorkspace(Eigen::Index batch_size, Eigen::Index T,
                           Eigen::Index n)
      : batch_size_(batch_size), T_(T), n_(n) {
    if (batch_size_ < 1 || T_ < 1 || n_ < 1) {
      throw std::invalid_argument(
          "Eigen workspace batch size and dimensions must be positive");
    }
    solvers_.reserve(batch_size_);
    for (Eigen::Index batch = 0; batch < batch_size_; ++batch) {
      solvers_.emplace_back(T_, n_);
    }
  }

  ScalarType scalar_type() const noexcept override {
    if constexpr (std::is_same_v<Scalar, float>) {
      return ScalarType::kFloat32;
    }
    return ScalarType::kFloat64;
  }

  bool matches(Eigen::Index batch_size, Eigen::Index T,
               Eigen::Index n) const noexcept override {
    return batch_size == batch_size_ && T == T_ && n == n_;
  }

  void Solve(const Scalar* blocks, const Scalar* rhs, bool factorize,
             Scalar* output, Eigen::Index batch_size, Eigen::Index T,
             Eigen::Index n) {
    if (batch_size != batch_size_ || T != T_ || n != n_) {
      throw std::invalid_argument(
          "Eigen workspace dimensions do not match solve buffers");
    }
    const Eigen::Index blocks_per_batch = T * n * 3 * n;
    const Eigen::Index values_per_batch = T * n;
    for (Eigen::Index batch = 0; batch < batch_size_; ++batch) {
      auto& solver = solvers_[batch];
      if (factorize) {
        solver.Factorize(blocks + batch * blocks_per_batch);
      }
      solver.Solve(rhs + batch * values_per_batch,
                   output + batch * values_per_batch);
    }
  }

private:
  Eigen::Index batch_size_;
  Eigen::Index T_;
  Eigen::Index n_;
  std::vector<BlockTridiSolver<Scalar>> solvers_;
};

std::mutex registry_mutex;
std::unordered_map<std::uint64_t, std::unique_ptr<WorkspaceEntry>> registry;
std::uint64_t next_handle = 1;

std::uint64_t CreateEmptyWorkspace() {
  auto entry = std::make_unique<WorkspaceEntry>();
  std::lock_guard<std::mutex> lock(registry_mutex);
  if (next_handle == 0) {
    throw std::overflow_error("Eigen workspace handle space is exhausted");
  }
  const std::uint64_t handle = next_handle++;
  registry.emplace(handle, std::move(entry));
  return handle;
}

WorkspaceEntry& FindEntry(std::uint64_t handle) {
  const auto position = registry.find(handle);
  if (position == registry.end()) {
    throw std::invalid_argument("Eigen workspace handle is not live");
  }
  return *position->second;
}

template <typename Scalar>
EigenBlockTridiWorkspace<Scalar>& EnsureCapacity(
    WorkspaceEntry& entry, Eigen::Index batch_size, Eigen::Index T,
    Eigen::Index n) {
  const ScalarType expected = std::is_same_v<Scalar, float>
                                  ? ScalarType::kFloat32
                                  : ScalarType::kFloat64;
  if (entry.capacity == nullptr ||
      entry.capacity->scalar_type() != expected ||
      !entry.capacity->matches(batch_size, T, n)) {
    entry.capacity = std::make_unique<EigenBlockTridiWorkspace<Scalar>>(batch_size, T, n);
  }
  return static_cast<EigenBlockTridiWorkspace<Scalar>&>(*entry.capacity);
}

template <typename Scalar>
void SolveOwned(std::uint64_t handle, const Scalar* blocks, const Scalar* rhs,
                bool factorize, Scalar* output, Eigen::Index batch_size,
                 Eigen::Index T, Eigen::Index n) {
  std::unique_lock<std::mutex> registry_lock(registry_mutex);
  auto& entry = FindEntry(handle);
  std::unique_lock<std::mutex> execution_lock(entry.execution_mutex);
  registry_lock.unlock();
  auto& workspace = EnsureCapacity<Scalar>(entry, batch_size, T, n);
  workspace.Solve(blocks, rhs, factorize, output, batch_size, T, n);
}

void DestroyWorkspace(std::uint64_t handle) {
  std::unique_ptr<WorkspaceEntry> entry;
  std::unique_lock<std::mutex> registry_lock(registry_mutex);
  const auto position = registry.find(handle);
  if (position == registry.end()) {
    throw std::invalid_argument("Eigen workspace handle is not live");
  }
  std::unique_lock<std::mutex> execution_lock(
      position->second->execution_mutex);
  entry = std::move(position->second);
  registry.erase(position);
  registry_lock.unlock();
  execution_lock.unlock();
  entry.reset();
}

void EnsureWorkspaceCapacity(std::uint64_t handle, Eigen::Index batch_size,
                             Eigen::Index T, Eigen::Index n,
                             int dtype_bytes) {
  std::unique_lock<std::mutex> registry_lock(registry_mutex);
  auto& entry = FindEntry(handle);
  std::unique_lock<std::mutex> execution_lock(entry.execution_mutex);
  registry_lock.unlock();
  if (dtype_bytes == 4) {
    EnsureCapacity<float>(entry, batch_size, T, n);
  } else if (dtype_bytes == 8) {
    EnsureCapacity<double>(entry, batch_size, T, n);
  } else {
    throw std::invalid_argument("workspace dtype must be 4 or 8 bytes");
  }
}

template <typename Scalar>
py::array_t<Scalar> SolvePy(
    std::uint64_t workspace_handle,
    py::array_t<Scalar, py::array::c_style> blocks,
    py::array_t<Scalar, py::array::c_style> rhs,
    bool factorize) {
  const auto block_info = blocks.request();
  const auto rhs_info = rhs.request();
  if ((block_info.ndim != 3 && block_info.ndim != 4) ||
      rhs_info.ndim != block_info.ndim - 1) {
    throw std::invalid_argument(
        "expected blocks (..., T, n, 3n) and rhs (..., T, n)");
  }

  const bool batched = block_info.ndim == 4;
  const Eigen::Index batch_size = batched ? block_info.shape[0] : 1;
  const Eigen::Index T = block_info.shape[batched ? 1 : 0];
  const Eigen::Index n = block_info.shape[batched ? 2 : 1];
  const Eigen::Index rhs_batch_size = batched ? rhs_info.shape[0] : 1;
  const Eigen::Index rhs_T = rhs_info.shape[batched ? 1 : 0];
  const Eigen::Index rhs_n = rhs_info.shape[batched ? 2 : 1];
  if (batch_size < 1 || T < 1 || n < 1 ||
      block_info.shape[batched ? 3 : 2] != 3 * n ||
      rhs_batch_size != batch_size || rhs_T != T || rhs_n != n) {
    throw std::invalid_argument(
        "expected blocks (..., T, n, 3n) and rhs (..., T, n)");
  }

  py::array_t<Scalar> output = batched ? py::array_t<Scalar>({batch_size, T, n})
                                       : py::array_t<Scalar>({T, n});
  SolveOwned(workspace_handle, static_cast<const Scalar*>(block_info.ptr),
             static_cast<const Scalar*>(rhs_info.ptr), factorize,
             output.mutable_data(), batch_size, T, n);
  return output;
}

template <typename Buffer, typename Scalar>
ffi::Error EigenBlockTridiCpuImpl(Buffer blocks, Buffer rhs, BufPred factorize,
                                  ffi::Result<Buffer> output,
                                  std::int64_t workspace_handle) {
  const auto block_dims = blocks.dimensions();
  const auto rhs_dims = rhs.dimensions();
  const auto output_dims = output->dimensions();
  if ((block_dims.size() != 3 && block_dims.size() != 4) ||
      rhs_dims.size() != block_dims.size() - 1 ||
      output_dims.size() != rhs_dims.size() || factorize.element_count() != 1) {
    return ffi::Error::InvalidArgument(
        "expected blocks (..., T, n, 3n), rhs/output (..., T, n), and one "
        "factorization flag");
  }

  const bool batched = block_dims.size() == 4;
  const Eigen::Index batch_size = batched ? block_dims[0] : 1;
  const Eigen::Index T = block_dims[batched ? 1 : 0];
  const Eigen::Index n = block_dims[batched ? 2 : 1];
  const Eigen::Index rhs_batch_size = batched ? rhs_dims[0] : 1;
  const Eigen::Index rhs_T = rhs_dims[batched ? 1 : 0];
  const Eigen::Index rhs_n = rhs_dims[batched ? 2 : 1];
  const Eigen::Index output_batch_size = batched ? output_dims[0] : 1;
  const Eigen::Index output_T = output_dims[batched ? 1 : 0];
  const Eigen::Index output_n = output_dims[batched ? 2 : 1];
  if (batch_size < 1 || T < 1 || n < 1 ||
      block_dims[batched ? 3 : 2] != 3 * n || rhs_batch_size != batch_size ||
      rhs_T != T || rhs_n != n || output_batch_size != batch_size ||
      output_T != T || output_n != n) {
    return ffi::Error::InvalidArgument(
        "expected blocks (..., T, n, 3n) and rhs/output (..., T, n)");
  }

  try {
    if (workspace_handle <= 0) {
      return ffi::Error::InvalidArgument(
          "Eigen workspace handle must be positive");
    }
    SolveOwned(static_cast<std::uint64_t>(workspace_handle),
               blocks.typed_data(), rhs.typed_data(), factorize.typed_data()[0],
               output->typed_data(), batch_size, T, n);
  } catch (const std::invalid_argument& error) {
    return ffi::Error::InvalidArgument(error.what());
  } catch (const std::exception& error) {
    return ffi::Error::Internal(error.what());
  }
  return ffi::Error::Success();
}

extern "C" XLA_FFI_Error* EigenBlockTridiCpu(XLA_FFI_CallFrame* call_frame) {
  static auto* handler =
      ffi::Ffi::Bind()
          .Arg<BufF32>()
          .Arg<BufF32>()
          .Arg<BufPred>()
          .Ret<BufF32>()
          .Attr<std::int64_t>("workspace_handle")
          .To(EigenBlockTridiCpuImpl<BufF32, float>)
          .release();
  return handler->Call(call_frame);
}

extern "C" XLA_FFI_Error* EigenBlockTridiCpuF64(XLA_FFI_CallFrame* call_frame) {
  static auto* handler =
      ffi::Ffi::Bind()
          .Arg<BufF64>()
          .Arg<BufF64>()
          .Arg<BufPred>()
          .Ret<BufF64>()
          .Attr<std::int64_t>("workspace_handle")
          .To(EigenBlockTridiCpuImpl<BufF64, double>)
          .release();
  return handler->Call(call_frame);
}

PYBIND11_MODULE(_eigen_block_tridi, module) {
  module.def("create_workspace", &CreateEmptyWorkspace);
  module.def("ensure_workspace_capacity", &EnsureWorkspaceCapacity);
  module.def("destroy_workspace", &DestroyWorkspace);
  module.def("solve_float", &SolvePy<float>);
  module.def("solve_double", &SolvePy<double>);
  module.def("registrations", []() {
    py::dict registrations;
    registrations["eigen_block_tridi_cpu"] =
        py::capsule(reinterpret_cast<void*>(EigenBlockTridiCpu));
    registrations["eigen_block_tridi_cpu_f64"] =
        py::capsule(reinterpret_cast<void*>(EigenBlockTridiCpuF64));
    return registrations;
  });
}
