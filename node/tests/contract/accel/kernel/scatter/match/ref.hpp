#pragma once

#include <kernel/program/compute/scatter/reference.hpp>

#include "../local.hpp"

#include <array>

namespace node_accel_contract::scatter::match {

template <typename T> struct Reference {
  std::array<T, 6u> expected{};
  bool ok = false;
};

template <typename T>
[[nodiscard]] Reference<T>
BuildReference(const rund::kernel::ScatterElement element,
               const std::array<T, 4u> &values,
               const std::array<rund::kernel::u32, 4u> &indices,
               const std::array<T, 6u> &initial_output) {
  Reference<T> out{.expected = initial_output};
  out.ok = element == rund::kernel::ScatterElement::U32
               ? rund::kernel::ReferenceScatterU32(
                     reinterpret_cast<const rund::kernel::u32 *>(values.data()),
                     indices.data(),
                     reinterpret_cast<rund::kernel::u32 *>(out.expected.data()),
                     indices.size(), out.expected.size())
                     .ok
               : rund::kernel::ReferenceScatterU64(
                     reinterpret_cast<const rund::kernel::u64 *>(values.data()),
                     indices.data(),
                     reinterpret_cast<rund::kernel::u64 *>(out.expected.data()),
                     indices.size(), out.expected.size())
                     .ok;
  return out;
}

} // namespace node_accel_contract::scatter::match
