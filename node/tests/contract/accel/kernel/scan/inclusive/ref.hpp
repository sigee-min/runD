#pragma once

#include <kernel/program/compute/scan/reference.hpp>

#include "../local.hpp"

#include <array>

namespace node_accel_contract::scan_inclusive {

template <typename T, std::size_t N> struct Reference {
  std::array<T, N> expected{};
  bool ok = false;
};

template <typename T, std::size_t N>
[[nodiscard]] Reference<T, N>
BuildReference(const rund::kernel::ScanElement element,
               const std::array<T, N> &input) {
  Reference<T, N> out{};
  rund::kernel::u64 total = 0u;
  out.ok = element == rund::kernel::ScanElement::U32
               ? rund::kernel::ReferenceInclusiveScanU32(
                     reinterpret_cast<const rund::kernel::u32 *>(input.data()),
                     reinterpret_cast<rund::kernel::u32 *>(out.expected.data()),
                     input.size(), &total)
                     .ok
               : rund::kernel::ReferenceInclusiveScanU64(
                     reinterpret_cast<const rund::kernel::u64 *>(input.data()),
                     reinterpret_cast<rund::kernel::u64 *>(out.expected.data()),
                     input.size(), &total)
                     .ok;
  return out;
}

} // namespace node_accel_contract::scan_inclusive
