#pragma once

#include <kernel/program/compute/model.hpp>

namespace rund::kernel {

enum class ScatterReduceOp : u8 {
  Sum = 1u,
  Min = 2u,
  Max = 3u,
};

struct ScatterReduceDesc final {
  ScatterReduceOp op = ScatterReduceOp::Sum;
  ComputeDomain domain = ComputeDomain::U32;
  ComputeFixedFormat fixed_format{};
  u64 element_count = 0u;
  u64 output_count = 0u;
  ComputeCountSource count_source = ComputeCountSource::Descriptor;
};

struct ScatterReducePlan final {
  ScatterReduceOp op = ScatterReduceOp::Sum;
  ComputeDomain domain = ComputeDomain::U32;
  ComputeFixedFormat fixed_format{};
  u64 element_count = 0u;
  u64 output_count = 0u;
  u64 element_bytes = 0u;
  u64 index_bytes = 4u;
  u64 sorted_index_bytes = 0u;
  u64 sorted_value_bytes = 0u;
  u64 segment_bytes = 0u;
  u64 status_bytes = 0u;
  u64 indirect_bytes = 0u;
  u64 temp_bytes = 0u;
  u32 radix_pass_count = 0u;
  u32 fold_pass_count = 0u;
  u32 pass_count = 0u;
  ComputeCountSource count_source = ComputeCountSource::Descriptor;
  bool ok = false;
  const char *reason = "compute_scatter_reduce_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

struct ScatterReduceHash final {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct ScatterReduceResult final {
  u64 element_count = 0u;
  u64 output_count = 0u;
  u64 first_rejected_ordinal = 0u;
  u64 conflict_count = 0u;
  bool ok = false;
  const char *reason = "compute_scatter_reduce_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

} // namespace rund::kernel
