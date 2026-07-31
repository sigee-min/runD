#pragma once

#include <kernel/core/model.hpp>
#include <kernel/program/compute/matrix/model.hpp>

namespace rund::kernel {

enum class FactorOp : u8 {
  LU = 1u,
  QR = 2u,
  Cholesky = 3u,
};

enum class FactorOutput : u8 {
  Packed = 1u,
  Separate = 2u,
};

enum class PivotOp : u8 {
  None = 1u,
  Partial = 2u,
};

enum class FactorStatus : u32 {
  Ok = 0u,
  Singular = 1u,
  NonSpd = 2u,
  PivotUnderflow = 3u,
  InvalidScaling = 4u,
};

struct FactorShape {
  MatrixLayout layout = MatrixLayout::RowMajor;
  u64 rows = 0u;
  u64 cols = 0u;
  u64 batch_count = 1u;
  u32 element_bytes = 4u;
};

struct FactorDesc {
  FactorOp op = FactorOp::LU;
  MatrixLayout layout = MatrixLayout::RowMajor;
  FactorOutput output = FactorOutput::Packed;
  PivotOp pivot = PivotOp::Partial;
  u64 rows = 0u;
  u64 cols = 0u;
  u64 batch_count = 1u;
  u32 element_bytes = 4u;
  ComputeFixedFormat fixed_format{};
};

struct FactorPlan {
  FactorOp op = FactorOp::LU;
  MatrixLayout layout = MatrixLayout::RowMajor;
  FactorOutput output = FactorOutput::Packed;
  PivotOp pivot = PivotOp::Partial;
  u64 rows = 0u;
  u64 cols = 0u;
  u64 batch_count = 0u;
  u64 input_count = 0u;
  u64 factor_count = 0u;
  u64 aux_count = 0u;
  u64 status_count = 0u;
  u64 workspace_bytes = 0u;
  u32 element_bytes = 0u;
  ComputeFixedFormat fixed_format{};
  u64 pass_count = 0u;
  bool ok = false;
  const char* reason = "compute_factor_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

struct FactorHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct FactorResult {
  u64 failed_batches = 0u;
  u64 first_failed_batch = 0u;
  FactorStatus first_status = FactorStatus::Ok;
  bool ok = false;
  const char* reason = "compute_factor_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

}  // namespace rund::kernel
