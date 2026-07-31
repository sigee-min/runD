#pragma once

#include <kernel/program/compute/gather/reference.hpp>

#include "../local.hpp"

#include <array>

namespace node_accel_contract::gather::match {

template <typename T> struct Reference {
  std::array<T, 4u> expected{};
  bool ok = false;
};

template <typename T>
[[nodiscard]] Reference<T>
BuildReference(const rund::kernel::GatherElement element,
               const std::array<T, 6u> &values,
               const std::array<rund::kernel::u32, 4u> &indices) {
  Reference<T> out{};
  out.ok = element == rund::kernel::GatherElement::U32
               ? rund::kernel::ReferenceGatherU32(
                     reinterpret_cast<const rund::kernel::u32 *>(values.data()),
                     indices.data(),
                     reinterpret_cast<rund::kernel::u32 *>(out.expected.data()),
                     indices.size(), values.size())
                     .ok
               : rund::kernel::ReferenceGatherU64(
                     reinterpret_cast<const rund::kernel::u64 *>(values.data()),
                     indices.data(),
                     reinterpret_cast<rund::kernel::u64 *>(out.expected.data()),
                     indices.size(), values.size())
                     .ok;
  return out;
}

} // namespace node_accel_contract::gather::match
