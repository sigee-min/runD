#pragma once

#include "scratch.hpp"

#include <kernel/program/compute/scatter.hpp>

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail {
using rund::kernel::ScatterResult;
namespace {

template <class Buffer>
[[nodiscard]] bool EnsureScatterCapacity(Buffer &buffer,
                                         const std::size_t cap) {
  if (buffer.size() >= cap) {
    return true;
  }
  if constexpr (requires { buffer.resize(cap); }) {
    buffer.resize(cap);
    return buffer.size() >= cap;
  }
  return false;
}

template <class Scratch>
[[nodiscard]] u32 NextScatterEpoch(Scratch &scratch) noexcept {
  u32 &epoch = scratch.scatter_epoch();
  if (epoch == std::numeric_limits<u32>::max()) {
    const std::span marks = scratch.scatter_marks();
    std::fill(marks.begin(), marks.end(), u32{0u});
    epoch = 1u;
    return epoch;
  }
  ++epoch;
  return epoch;
}

template <class Scratch>
[[nodiscard]] bool MarkScatterTarget(Scratch &scratch, const std::size_t cap,
                                     const u32 epoch,
                                     const u32 target) noexcept {
  const std::size_t mask = cap - 1u;
  std::size_t slot = (static_cast<std::size_t>(target) * 2654435761u) & mask;
  while (scratch.marks[slot] == epoch) {
    if (scratch.keys[slot] == target) {
      return false;
    }
    slot = (slot + 1u) & mask;
  }
  scratch.marks[slot] = epoch;
  scratch.keys[slot] = target;
  return true;
}

[[nodiscard]] inline ScatterResult
RejectScatter(const u64 element_count, const u64 output_count,
              const u64 first_rejected_index,
              const char *const reason) noexcept {
  return ScatterResult{
      .element_count = element_count,
      .output_count = output_count,
      .first_rejected_index = first_rejected_index,
      .reason = reason,
  };
}

template <typename Scratch, typename Element>
[[nodiscard]] ScatterResult
ExecuteLinearScatter(Scratch &scratch, const Element *const values,
                     const u32 *const indices, Element *const output,
                     const u64 element_count, const u64 output_count,
                     const std::size_t scratch_slots) {
  if (scratch_slots == 0u) {
    return RejectScatter(element_count, output_count, 0u,
                         "compute_scatter_temp_overflow");
  }
  if (!scratch.scatter_ready() ||
      !EnsureScatterCapacity(scratch.keys, scratch_slots) ||
      !EnsureScatterCapacity(scratch.marks, scratch_slots)) {
    return RejectScatter(element_count, output_count, 0u,
                         "compute_scatter_temp_overflow");
  }
  const u32 epoch = NextScatterEpoch(scratch);

  for (u64 index = 0u; index < element_count; ++index) {
    const u32 target = indices[index];
    if (static_cast<u64>(target) >= output_count) {
      return RejectScatter(element_count, output_count, index,
                           "compute_scatter_index_out_of_range");
    }
    if (!MarkScatterTarget(scratch, scratch_slots, epoch, target)) {
      return RejectScatter(element_count, output_count, index,
                           "compute_scatter_duplicate_index");
    }
  }

  for (u64 index = 0u; index < element_count; ++index) {
    output[indices[index]] = values[index];
  }
  return ScatterResult{
      .element_count = element_count,
      .output_count = output_count,
      .first_rejected_index = element_count,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace
} // namespace rund::node::accel::detail
