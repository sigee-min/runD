#pragma once

#include <kernel/program/compute/stencil/reference.hpp>

#include "../local.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <type_traits>

namespace node_accel_contract::stencil::match {

template <typename T, std::size_t Count> struct Reference {
  std::array<T, Count> expected{};
  bool ok = false;
};

template <typename T, std::size_t Count>
[[nodiscard]] Reference<T, Count>
BuildReference(const rund::kernel::StencilOp op,
               const rund::kernel::StencilElement element,
               const rund::kernel::u64 radius,
               const std::array<T, Count> &input, const bool signed_domain) {
  Reference<T, Count> out{};
  const bool u32 = element == rund::kernel::StencilElement::U32;
  if (signed_domain) {
    using S = std::conditional_t<sizeof(T) == sizeof(rund::kernel::i32),
                                 rund::kernel::i32, rund::kernel::i64>;
    using U = std::make_unsigned_t<S>;
    const auto *values = reinterpret_cast<const S *>(input.data());
    auto *output = reinterpret_cast<S *>(out.expected.data());
    for (std::size_t index = 0; index < Count; ++index) {
      S value = values[index];
      U sum = std::bit_cast<U>(value);
      for (std::size_t distance = 1; distance <= radius; ++distance) {
        const std::size_t left = index < distance ? 0 : index - distance;
        const std::size_t right = std::min(index + distance, Count - 1u);
        if (op == rund::kernel::StencilOp::Min) {
          value = std::min(value, std::min(values[left], values[right]));
        } else if (op == rund::kernel::StencilOp::Max) {
          value = std::max(value, std::max(values[left], values[right]));
        } else {
          sum += std::bit_cast<U>(values[left]);
          sum += std::bit_cast<U>(values[right]);
        }
      }
      output[index] =
          op == rund::kernel::StencilOp::Sum ? std::bit_cast<S>(sum) : value;
    }
    out.ok = true;
    return out;
  }
  if (op == rund::kernel::StencilOp::Sum) {
    out.ok =
        u32 ? rund::kernel::ReferenceStencilSumU32(
                  reinterpret_cast<const rund::kernel::u32 *>(input.data()),
                  reinterpret_cast<rund::kernel::u32 *>(out.expected.data()),
                  input.size(), radius)
                  .ok
            : rund::kernel::ReferenceStencilSumU64(
                  reinterpret_cast<const rund::kernel::u64 *>(input.data()),
                  reinterpret_cast<rund::kernel::u64 *>(out.expected.data()),
                  input.size(), radius)
                  .ok;
  } else if (op == rund::kernel::StencilOp::Min) {
    out.ok =
        u32 ? rund::kernel::ReferenceStencilMinU32(
                  reinterpret_cast<const rund::kernel::u32 *>(input.data()),
                  reinterpret_cast<rund::kernel::u32 *>(out.expected.data()),
                  input.size(), radius)
                  .ok
            : rund::kernel::ReferenceStencilMinU64(
                  reinterpret_cast<const rund::kernel::u64 *>(input.data()),
                  reinterpret_cast<rund::kernel::u64 *>(out.expected.data()),
                  input.size(), radius)
                  .ok;
  } else if (op == rund::kernel::StencilOp::Max) {
    out.ok =
        u32 ? rund::kernel::ReferenceStencilMaxU32(
                  reinterpret_cast<const rund::kernel::u32 *>(input.data()),
                  reinterpret_cast<rund::kernel::u32 *>(out.expected.data()),
                  input.size(), radius)
                  .ok
            : rund::kernel::ReferenceStencilMaxU64(
                  reinterpret_cast<const rund::kernel::u64 *>(input.data()),
                  reinterpret_cast<rund::kernel::u64 *>(out.expected.data()),
                  input.size(), radius)
                  .ok;
  }
  return out;
}

} // namespace node_accel_contract::stencil::match
