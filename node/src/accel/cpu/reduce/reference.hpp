#pragma once

#include <kernel/program/compute/reduce/reference.hpp>

#include <limits>

namespace rund::node::accel::detail {

template <class S>
[[nodiscard]] inline rund::kernel::ReduceResult
ReferenceReduceSigned(const rund::kernel::ReduceOp op, const S *const input,
                      S *const output,
                      const rund::kernel::u64 element_count) noexcept {
  if (input == nullptr || output == nullptr || element_count == 0u) {
    return rund::kernel::ReduceResult{.reason =
                                          "compute_reduce_buffer_invalid"};
  }
  S value = op == rund::kernel::ReduceOp::Min   ? std::numeric_limits<S>::max()
            : op == rund::kernel::ReduceOp::Max ? std::numeric_limits<S>::min()
                                                : S{0};
  for (rund::kernel::u64 index = 0u; index < element_count; ++index) {
    if (op == rund::kernel::ReduceOp::Sum) {
      if (__builtin_add_overflow(value, input[index], &value)) {
        return rund::kernel::ReduceResult{.element_count = element_count,
                                          .reason =
                                              "compute_reduce_sum_overflow"};
      }
    } else if (op == rund::kernel::ReduceOp::CountNonzero) {
      if (input[index] != 0 && __builtin_add_overflow(value, S{1}, &value)) {
        return rund::kernel::ReduceResult{.element_count = element_count,
                                          .reason =
                                              "compute_reduce_count_overflow"};
      }
    } else if (op == rund::kernel::ReduceOp::Min) {
      value = input[index] < value ? input[index] : value;
    } else {
      value = input[index] > value ? input[index] : value;
    }
  }
  *output = value;
  return rund::kernel::ReduceResult{
      .element_count = element_count, .ok = true, .reason = "ok"};
}

[[nodiscard]] inline rund::kernel::ReduceResult
ReferenceReduceU32(const rund::kernel::ReduceOp op,
                   const rund::kernel::u32 *const input,
                   rund::kernel::u32 *const output,
                   const rund::kernel::u64 element_count) noexcept {
  if (op == rund::kernel::ReduceOp::CountNonzero) {
    return rund::kernel::ReferenceReduceCountNonzeroU32(input, output,
                                                        element_count);
  }
  if (op == rund::kernel::ReduceOp::Min) {
    return rund::kernel::ReferenceReduceMinU32(input, output, element_count);
  }
  if (op == rund::kernel::ReduceOp::Max) {
    return rund::kernel::ReferenceReduceMaxU32(input, output, element_count);
  }
  return rund::kernel::ReferenceReduceSumU32(input, output, element_count);
}

[[nodiscard]] inline rund::kernel::ReduceResult
ReferenceReduceU64(const rund::kernel::ReduceOp op,
                   const rund::kernel::u64 *const input,
                   rund::kernel::u64 *const output,
                   const rund::kernel::u64 element_count) noexcept {
  if (op == rund::kernel::ReduceOp::CountNonzero) {
    return rund::kernel::ReferenceReduceCountNonzeroU64(input, output,
                                                        element_count);
  }
  if (op == rund::kernel::ReduceOp::Min) {
    return rund::kernel::ReferenceReduceMinU64(input, output, element_count);
  }
  if (op == rund::kernel::ReduceOp::Max) {
    return rund::kernel::ReferenceReduceMaxU64(input, output, element_count);
  }
  return rund::kernel::ReferenceReduceSumU64(input, output, element_count);
}

} // namespace rund::node::accel::detail
