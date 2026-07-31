#pragma once

#include "../local.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <span>
#include <type_traits>

namespace node_accel_contract::reduce::match {

template <typename T> struct Reference {
  T expected{};
  bool ok{false};
};

} // namespace node_accel_contract::reduce::match

#include "ref/u32.hpp"
#include "ref/u64.hpp"

namespace node_accel_contract::reduce::match {

template <typename T>
[[nodiscard]] Reference<T>
BuildReference(const rund::kernel::ReduceOp op,
               const rund::kernel::ReduceElement element,
               const std::span<const T> input, const bool signed_domain) {
  if (signed_domain) {
    using S = std::conditional_t<sizeof(T) == sizeof(rund::kernel::i32),
                                 rund::kernel::i32, rund::kernel::i64>;
    const auto *values = reinterpret_cast<const S *>(input.data());
    if (op == rund::kernel::ReduceOp::Sum ||
        op == rund::kernel::ReduceOp::CountNonzero) {
      rund::kernel::i128 total = 0;
      for (std::size_t index = 0u; index < input.size(); ++index) {
        total += op == rund::kernel::ReduceOp::CountNonzero
                     ? (values[index] != 0 ? 1 : 0)
                     : static_cast<rund::kernel::i128>(values[index]);
      }
      if (total < static_cast<rund::kernel::i128>(std::numeric_limits<S>::min()) ||
          total > static_cast<rund::kernel::i128>(std::numeric_limits<S>::max())) {
        return Reference<T>{};
      }
      return Reference<T>{
          .expected = std::bit_cast<T>(static_cast<S>(total)), .ok = true};
    }
    S result = values[0];
    for (std::size_t index = 1u; index < input.size(); ++index) {
      if (op == rund::kernel::ReduceOp::Min) {
        result = std::min(result, values[index]);
      } else if (op == rund::kernel::ReduceOp::Max) {
        result = std::max(result, values[index]);
      }
    }
    return Reference<T>{.expected = std::bit_cast<T>(result), .ok = true};
  }
  if (element == rund::kernel::ReduceElement::U32) {
    return BuildReferenceU32(op, input);
  }
  return BuildReferenceU64(op, input);
}

} // namespace node_accel_contract::reduce::match
