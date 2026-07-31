#pragma once

#include <kernel/program/compute/model.hpp>

namespace rund::kernel {

enum class ReduceOp : u8 {
  Sum = 1u,
  CountNonzero = 2u,
  Min = 3u,
  Max = 4u,
};

enum class ReduceElement : u8 {
  U32 = 1u,
  U64 = 2u,
};

inline constexpr u64 kReduceItemsPerThread = 8u;
inline constexpr u64 kReduceFirstPassMaxGroups = 128u;
inline constexpr u64 kReduceWideElementBytes = 16u;
inline constexpr u64 kReduceNarrowChunkItems = 256u;

[[nodiscard]] constexpr bool
ReduceUsesWidePartials(const ReduceOp op) noexcept {
  return op == ReduceOp::Sum || op == ReduceOp::CountNonzero;
}

struct ReduceDesc {
  ReduceOp op = ReduceOp::Sum;
  ReduceElement element = ReduceElement::U32;
  u64 element_count = 0u;
  u64 block_size = 0u;
  ComputeCountSource count_source = ComputeCountSource::Descriptor;
};

struct ReducePlan {
  ReduceOp op = ReduceOp::Sum;
  ReduceElement element = ReduceElement::U32;
  u64 element_count = 0u;
  u64 element_bytes = 0u;
  u64 block_size = 0u;
  u64 items_per_thread = 0u;
  u64 first_pass_group_count = 0u;
  u64 pass_count = 0u;
  u64 partial_element_count = 0u;
  u64 partial_element_bytes = 0u;
  u64 partial_bytes = 0u;
  u64 status_bytes = 0u;
  u64 temp_bytes = 0u;
  ComputeCountSource count_source = ComputeCountSource::Descriptor;
  bool ok = false;
  const char* reason = "compute_reduce_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

struct ReduceHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct ReduceResult {
  u64 element_count = 0u;
  u64 total = 0u;
  bool ok = false;
  const char* reason = "compute_reduce_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

}  // namespace rund::kernel
