#pragma once

#include <kernel/program/compute/scan/reference.hpp>

#include "../local.hpp"

#include <array>

namespace node_accel_contract::collective {

template <typename T>
[[nodiscard]] bool BuildScanReference(const rund::kernel::ScanElement element,
                                      const std::array<T, 8u> &input,
                                      std::array<T, 8u> &expected) {
  std::uint64_t total = 0u;
  return element == rund::kernel::ScanElement::U32
             ? rund::kernel::ReferenceExclusiveScanU32(
                   reinterpret_cast<const rund::kernel::u32 *>(input.data()),
                   reinterpret_cast<rund::kernel::u32 *>(expected.data()),
                   input.size(), &total)
                   .ok
             : rund::kernel::ReferenceExclusiveScanU64(
                   reinterpret_cast<const rund::kernel::u64 *>(input.data()),
                   reinterpret_cast<rund::kernel::u64 *>(expected.data()),
                   input.size(), &total)
                   .ok;
}

} // namespace node_accel_contract::collective
