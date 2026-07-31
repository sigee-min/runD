#pragma once

#include <kernel/program/compute/reduce/reference.hpp>

namespace node_accel_contract::reduce::match {

template <typename T>
[[nodiscard]] Reference<T> BuildReferenceU32(const rund::kernel::ReduceOp op,
                                             const std::span<const T> input) {
  Reference<T> ref{};
  const auto *const values =
      reinterpret_cast<const rund::kernel::u32 *>(input.data());
  auto *const expected = reinterpret_cast<rund::kernel::u32 *>(&ref.expected);
  if (op == rund::kernel::ReduceOp::CountNonzero) {
    ref.ok = rund::kernel::ReferenceReduceCountNonzeroU32(values, expected,
                                                          input.size())
                 .ok;
  } else if (op == rund::kernel::ReduceOp::Min) {
    ref.ok =
        rund::kernel::ReferenceReduceMinU32(values, expected, input.size()).ok;
  } else if (op == rund::kernel::ReduceOp::Max) {
    ref.ok =
        rund::kernel::ReferenceReduceMaxU32(values, expected, input.size()).ok;
  } else {
    ref.ok =
        rund::kernel::ReferenceReduceSumU32(values, expected, input.size()).ok;
  }
  return ref;
}

} // namespace node_accel_contract::reduce::match
