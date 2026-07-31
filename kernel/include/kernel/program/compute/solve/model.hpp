#pragma once

#include <kernel/core/model.hpp>
#include <kernel/program/compute/factor/model.hpp>
#include <kernel/program/compute/matrix/model.hpp>

namespace rund::kernel {

enum class SolveOp : u8 {
  Linear = 1u,
};

enum class SolveInput : u8 {
  Matrix = 1u,
  Factor = 2u,
};

enum class SolveStatus : u32 {
  Ok = 0u,
  Singular = 1u,
  NonSpd = 2u,
  PivotUnderflow = 3u,
  InvalidScaling = 4u,
};

struct SolveShape {
  MatrixLayout layout = MatrixLayout::RowMajor;
  u64 rows = 0u;
  u64 rhs_cols = 0u;
  u64 batch_count = 1u;
  u32 element_bytes = 4u;
};

struct SolveDesc {
  SolveOp op = SolveOp::Linear;
  SolveInput input = SolveInput::Matrix;
  FactorOp factor = FactorOp::LU;
  MatrixLayout layout = MatrixLayout::RowMajor;
  PivotOp pivot = PivotOp::Partial;
  u64 rows = 0u;
  u64 rhs_cols = 0u;
  u64 batch_count = 1u;
  u32 element_bytes = 4u;
  ComputeFixedFormat fixed_format{};
};

struct SolvePlan {
  SolveOp op = SolveOp::Linear;
  SolveInput input = SolveInput::Matrix;
  FactorOp factor = FactorOp::LU;
  MatrixLayout layout = MatrixLayout::RowMajor;
  PivotOp pivot = PivotOp::Partial;
  u64 rows = 0u;
  u64 rhs_cols = 0u;
  u64 batch_count = 0u;
  u64 matrix_count = 0u;
  u64 factor_count = 0u;
  u64 rhs_count = 0u;
  u64 output_count = 0u;
  u64 aux_count = 0u;
  u64 status_count = 0u;
  u64 workspace_bytes = 0u;
  u32 element_bytes = 0u;
  ComputeFixedFormat fixed_format{};
  u64 pass_count = 0u;
  bool ok = false;
  const char* reason = "compute_solve_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

struct SolveHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct SolveResult {
  u64 failed_batches = 0u;
  u64 first_failed_batch = 0u;
  SolveStatus first_status = SolveStatus::Ok;
  bool ok = false;
  const char* reason = "compute_solve_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

}  // namespace rund::kernel
