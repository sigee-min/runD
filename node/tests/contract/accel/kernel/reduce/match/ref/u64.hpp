#pragma once

#include <kernel/program/compute/reduce/reference.hpp>

namespace node_accel_contract::reduce::match {

template <typename T>
[[nodiscard]] Reference<T> BuildReferenceU64(const rund::kernel::ReduceOp op,
                                             const std::span<const T> input) {
  Reference<T> ref{};
  const auto *const values =
      reinterpret_cast<const rund::kernel::u64 *>(input.data());
  auto *const expected = reinterpret_cast<rund::kernel::u64 *>(&ref.expected);
  if (op == rund::kernel::ReduceOp::CountNonzero) {
    ref.ok = rund::kernel::ReferenceReduceCountNonzeroU64(values, expected,
                                                          input.size())
                 .ok;
  } else if (op == rund::kernel::ReduceOp::Min) {
    ref.ok =
        rund::kernel::ReferenceReduceMinU64(values, expected, input.size()).ok;
  } else if (op == rund::kernel::ReduceOp::Max) {
    ref.ok =
        rund::kernel::ReferenceReduceMaxU64(values, expected, input.size()).ok;
  } else {
    ref.ok =
        rund::kernel::ReferenceReduceSumU64(values, expected, input.size()).ok;
  }
  return ref;
}

} // namespace node_accel_contract::reduce::match
