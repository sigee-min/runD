#pragma once

#include <kernel/program/compute/reduce/model.hpp>

namespace rund::kernel {

struct SegmentedReduceDesc {
  ReduceOp op = ReduceOp::Sum;
  ReduceElement element = ReduceElement::U32;
  u64 element_count = 0u;
  u64 block_size = 0u;
};

struct SegmentedReducePlan {
  ReduceOp op = ReduceOp::Sum;
  ReduceElement element = ReduceElement::U32;
  u64 element_count = 0u;
  u64 element_bytes = 0u;
  u64 head_bytes = 0u;
  u64 block_size = 0u;
  u64 block_count = 0u;
  u64 pass_count = 0u;
  u64 temp_value_bytes = 0u;
  u64 temp_head_bytes = 0u;
  u64 status_bytes = 0u;
  u64 temp_bytes = 0u;
  bool ok = false;
  const char* reason = "compute_segmented_reduce_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

struct SegmentedReduceHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct SegmentedReduceResult {
  u64 element_count = 0u;
  u64 segment_count = 0u;
  u64 final_segment_total = 0u;
  bool ok = false;
  const char* reason = "compute_segmented_reduce_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

}  // namespace rund::kernel
