#pragma once

#include <kernel/core/model.hpp>
#include <kernel/program/compute/model.hpp>

namespace rund::kernel {

enum class TransformOp : u8 {
  Fourier = 1u,
};

enum class TransformDir : u8 {
  Forward = 1u,
  Inverse = 2u,
};

enum class TransformLayout : u8 {
  Split = 1u,
  Interleaved = 2u,
};

enum class TransformNorm : u8 {
  None = 1u,
  InverseLength = 2u,
  Unitary = 3u,
};

struct TransformDesc {
  TransformOp op = TransformOp::Fourier;
  TransformDir direction = TransformDir::Forward;
  TransformLayout layout = TransformLayout::Split;
  TransformNorm normalization = TransformNorm::None;
  u64 element_count = 0u;
  ComputeFixedFormat fixed_format{};
};

struct TransformPlan {
  TransformOp op = TransformOp::Fourier;
  TransformDir direction = TransformDir::Forward;
  TransformLayout layout = TransformLayout::Split;
  TransformNorm normalization = TransformNorm::None;
  u64 element_count = 0u;
  u64 twiddle_count = 0u;
  u64 workspace_bytes = 0u;
  u64 pass_count = 0u;
  u64 normalization_divisor = 0u;
  u32 element_bytes = 0u;
  ComputeFixedFormat fixed_format{};
  bool ok = false;
  const char *reason = "compute_transform_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

struct TransformHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct TransformResult {
  u64 element_count = 0u;
  bool ok = false;
  const char *reason = "compute_transform_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

} // namespace rund::kernel
