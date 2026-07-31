#pragma once

#include <kernel/core/model.hpp>
#include <kernel/program/compute/matrix/model.hpp>

namespace rund::kernel {

enum class SpectrumOp : u8 {
  SVD = 1u,
  Eigen = 2u,
};

enum class SpectrumDomain : u8 {
  SymmetricReal = 1u,
  GeneralReal = 2u,
};

enum class SpectrumVectors : u8 {
  None = 1u,
  ValuesOnly = 2u,
  Thin = 3u,
  Full = 4u,
};

enum class SpectrumStatus : u32 {
  Ok = 0u,
  NonConvergence = 1u,
  InvalidScaling = 2u,
};

struct SpectrumShape {
  MatrixLayout layout = MatrixLayout::RowMajor;
  u64 rows = 0u;
  u64 cols = 0u;
  u64 batch_count = 1u;
  u32 max_iterations = 32u;
  u32 element_bytes = 4u;
};

struct SpectrumDesc {
  SpectrumOp op = SpectrumOp::SVD;
  SpectrumDomain domain = SpectrumDomain::GeneralReal;
  SpectrumVectors vectors = SpectrumVectors::ValuesOnly;
  MatrixLayout layout = MatrixLayout::RowMajor;
  u64 rows = 0u;
  u64 cols = 0u;
  u64 batch_count = 1u;
  u32 max_iterations = 32u;
  u32 element_bytes = 4u;
  ComputeFixedFormat fixed_format{};
};

struct SpectrumPlan {
  SpectrumOp op = SpectrumOp::SVD;
  SpectrumDomain domain = SpectrumDomain::GeneralReal;
  SpectrumVectors vectors = SpectrumVectors::ValuesOnly;
  MatrixLayout layout = MatrixLayout::RowMajor;
  u64 rows = 0u;
  u64 cols = 0u;
  u64 batch_count = 0u;
  u64 input_count = 0u;
  u64 value_count = 0u;
  u64 vector_count = 0u;
  u64 status_count = 0u;
  u64 workspace_bytes = 0u;
  u32 max_iterations = 0u;
  u32 element_bytes = 0u;
  ComputeFixedFormat fixed_format{};
  u64 pass_count = 0u;
  bool ok = false;
  const char* reason = "compute_spectrum_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

struct SpectrumHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct SpectrumResult {
  u64 failed_batches = 0u;
  u64 first_failed_batch = 0u;
  SpectrumStatus first_status = SpectrumStatus::Ok;
  bool ok = false;
  const char* reason = "compute_spectrum_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

}  // namespace rund::kernel
