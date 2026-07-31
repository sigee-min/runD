#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/segmented/heads.hpp>
#include <kernel/program/compute/segmented/reduce/model.hpp>

namespace rund::kernel::segmented_reduce_reference_detail {

[[nodiscard]] constexpr SegmentedReduceResult
Reject(const u64 element_count, const u64 segment_count,
       const u64 final_segment_total, const char *const reason) noexcept {
  return SegmentedReduceResult{
      .element_count = element_count,
      .segment_count = segment_count,
      .final_segment_total = final_segment_total,
      .reason = reason,
  };
}

[[nodiscard]] inline SegmentedReduceResult
RejectPriority(const u32 *const heads, const u64 unread,
               const u64 element_count, const u64 segment_count,
               const u64 total, const char *const numeric_reason) noexcept {
  return !SegmentHeadsValid(heads, unread, element_count)
             ? Reject(element_count, segment_count, total,
                      "compute_segmented_reduce_segment_invalid")
             : Reject(element_count, segment_count, total, numeric_reason);
}

struct StepResult {
  u64 value = 0u;
  const char *reason = nullptr;
};

[[nodiscard]] constexpr StepResult Ok(const u64 value) noexcept {
  return StepResult{.value = value};
}

[[nodiscard]] constexpr StepResult Fail(const u64 value,
                                        const char *const reason) noexcept {
  return StepResult{.value = value, .reason = reason};
}

[[nodiscard]] inline bool StoreU32OrReject(const u64 element_count,
                                           const u64 segment_count,
                                           const u64 value, u32 *const out,
                                           SegmentedReduceResult *const result,
                                           const char *const reason) noexcept {
  if (value <= static_cast<u64>(~u32{0u})) {
    *out = static_cast<u32>(value);
    return true;
  }
  *result = Reject(element_count, segment_count, value, reason);
  return false;
}

[[nodiscard]] constexpr StepResult SumOrFail(const u64 lhs, const u64 rhs,
                                             const u64 max_value) noexcept {
  if (!checked::add(lhs, rhs)) {
    return Fail(lhs, "compute_segmented_reduce_sum_overflow");
  }
  const u64 sum = lhs + rhs;
  if (sum > max_value) {
    return Fail(sum, "compute_segmented_reduce_sum_overflow");
  }
  return Ok(sum);
}

template <typename Value, typename Seed, typename Step, typename Store>
[[nodiscard]] inline SegmentedReduceResult
ReferenceSegmentedReduce(const Value *const input,
                         const u32 *const segment_heads, Value *const output,
                         const u64 element_count, Seed seed, Step step,
                         Store store) noexcept {
  if (element_count == 0u) {
    return Reject(0u, 0u, 0u, "compute_segmented_reduce_count_zero");
  }
  if (input == nullptr || segment_heads == nullptr || output == nullptr) {
    return Reject(element_count, 0u, 0u,
                  "compute_segmented_reduce_buffer_invalid");
  }
  if (!SegmentHeadValid(segment_heads[0], 0u)) {
    return Reject(element_count, 0u, 0u,
                  "compute_segmented_reduce_segment_invalid");
  }

  StepResult running{};
  u64 segment_count = 0u;
  for (u64 index = 0u; index < element_count; ++index) {
    const u32 head = segment_heads[index];
    if (!SegmentHeadValid(head, index)) {
      return Reject(element_count, segment_count, running.value,
                    "compute_segmented_reduce_segment_invalid");
    }
    if (head == 1u) {
      if (segment_count != 0u) {
        SegmentedReduceResult rejected{};
        if (!store(element_count, segment_count, running.value,
                   output + segment_count - 1u, &rejected)) {
          return RejectPriority(segment_heads, index + 1u, element_count,
                                segment_count, running.value, rejected.reason);
        }
      }
      ++segment_count;
      running = seed(static_cast<u64>(input[index]));
    } else {
      running = step(running.value, static_cast<u64>(input[index]));
    }
    if (running.reason != nullptr) {
      return RejectPriority(segment_heads, index + 1u, element_count,
                            segment_count, running.value, running.reason);
    }
  }

  SegmentedReduceResult rejected{};
  if (!store(element_count, segment_count, running.value,
             output + segment_count - 1u, &rejected)) {
    return rejected;
  }
  return SegmentedReduceResult{
      .element_count = element_count,
      .segment_count = segment_count,
      .final_segment_total = running.value,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace rund::kernel::segmented_reduce_reference_detail
