#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/reduce/model.hpp>

namespace rund::kernel {
namespace reduce_reference_detail {

[[nodiscard]] constexpr ReduceResult Reject(const u64 element_count,
                                            const u64 total,
                                            const char *const reason) noexcept {
  return ReduceResult{
      .element_count = element_count,
      .total = total,
      .reason = reason,
  };
}

template <typename T>
[[nodiscard]] inline ReduceResult
ReduceMinMax(const T *const input, T *const output, const u64 element_count,
             const bool maximum) noexcept {
  if (element_count == 0u) {
    return Reject(0u, 0u, "compute_reduce_count_zero");
  }
  if (input == nullptr || output == nullptr) {
    return Reject(element_count, 0u, "compute_reduce_buffer_invalid");
  }
  T selected = input[0u];
  for (u64 index = 1u; index < element_count; ++index) {
    const T value = input[index];
    if ((maximum && value > selected) || (!maximum && value < selected)) {
      selected = value;
    }
  }
  *output = selected;
  return ReduceResult{
      .element_count = element_count,
      .total = static_cast<u64>(selected),
      .ok = true,
      .reason = "ok",
  };
}

} // namespace reduce_reference_detail

[[nodiscard]] inline ReduceResult
ReferenceReduceSumU32(const u32 *const input, u32 *const output,
                      const u64 element_count) noexcept {
  if (element_count == 0u) {
    return reduce_reference_detail::Reject(0u, 0u, "compute_reduce_count_zero");
  }
  if (input == nullptr || output == nullptr) {
    return reduce_reference_detail::Reject(element_count, 0u,
                                           "compute_reduce_buffer_invalid");
  }

  u64 running = 0u;
  for (u64 index = 0u; index < element_count; ++index) {
    const u64 value = static_cast<u64>(input[index]);
    if (!checked::add(running, value)) {
      return reduce_reference_detail::Reject(element_count, running,
                                             "compute_reduce_sum_overflow");
    }
    running += value;
    if (running > static_cast<u64>(~u32{0u})) {
      return reduce_reference_detail::Reject(element_count, running,
                                             "compute_reduce_sum_overflow");
    }
  }

  *output = static_cast<u32>(running);
  return ReduceResult{
      .element_count = element_count,
      .total = running,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] inline ReduceResult
ReferenceReduceSumU64(const u64 *const input, u64 *const output,
                      const u64 element_count) noexcept {
  if (element_count == 0u) {
    return reduce_reference_detail::Reject(0u, 0u, "compute_reduce_count_zero");
  }
  if (input == nullptr || output == nullptr) {
    return reduce_reference_detail::Reject(element_count, 0u,
                                           "compute_reduce_buffer_invalid");
  }

  u64 running = 0u;
  for (u64 index = 0u; index < element_count; ++index) {
    if (!checked::add(running, input[index])) {
      return reduce_reference_detail::Reject(element_count, running,
                                             "compute_reduce_sum_overflow");
    }
    running += input[index];
  }

  *output = running;
  return ReduceResult{
      .element_count = element_count,
      .total = running,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] inline ReduceResult
ReferenceReduceCountNonzeroU32(const u32 *const input, u32 *const output,
                               const u64 element_count) noexcept {
  if (element_count == 0u) {
    return reduce_reference_detail::Reject(0u, 0u, "compute_reduce_count_zero");
  }
  if (input == nullptr || output == nullptr) {
    return reduce_reference_detail::Reject(element_count, 0u,
                                           "compute_reduce_buffer_invalid");
  }

  u64 running = 0u;
  for (u64 index = 0u; index < element_count; ++index) {
    if (input[index] != 0u) {
      ++running;
      if (running > static_cast<u64>(~u32{0u})) {
        return reduce_reference_detail::Reject(element_count, running,
                                               "compute_reduce_count_overflow");
      }
    }
  }

  *output = static_cast<u32>(running);
  return ReduceResult{
      .element_count = element_count,
      .total = running,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] inline ReduceResult
ReferenceReduceCountNonzeroU64(const u64 *const input, u64 *const output,
                               const u64 element_count) noexcept {
  if (element_count == 0u) {
    return reduce_reference_detail::Reject(0u, 0u, "compute_reduce_count_zero");
  }
  if (input == nullptr || output == nullptr) {
    return reduce_reference_detail::Reject(element_count, 0u,
                                           "compute_reduce_buffer_invalid");
  }

  u64 running = 0u;
  for (u64 index = 0u; index < element_count; ++index) {
    if (input[index] != 0u) {
      if (!checked::add(running, 1u)) {
        return reduce_reference_detail::Reject(element_count, running,
                                               "compute_reduce_count_overflow");
      }
      ++running;
    }
  }

  *output = running;
  return ReduceResult{
      .element_count = element_count,
      .total = running,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] inline ReduceResult
ReferenceReduceMinU32(const u32 *const input, u32 *const output,
                      const u64 element_count) noexcept {
  return reduce_reference_detail::ReduceMinMax(input, output, element_count,
                                               false);
}

[[nodiscard]] inline ReduceResult
ReferenceReduceMaxU32(const u32 *const input, u32 *const output,
                      const u64 element_count) noexcept {
  return reduce_reference_detail::ReduceMinMax(input, output, element_count,
                                               true);
}

[[nodiscard]] inline ReduceResult
ReferenceReduceMinU64(const u64 *const input, u64 *const output,
                      const u64 element_count) noexcept {
  return reduce_reference_detail::ReduceMinMax(input, output, element_count,
                                               false);
}

[[nodiscard]] inline ReduceResult
ReferenceReduceMaxU64(const u64 *const input, u64 *const output,
                      const u64 element_count) noexcept {
  return reduce_reference_detail::ReduceMinMax(input, output, element_count,
                                               true);
}

} // namespace rund::kernel
