#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace rund::node::accel::detail {

struct KernelViewSlot final {
  std::uint64_t binding{};
  std::size_t slot{std::numeric_limits<std::size_t>::max()};
  std::uint64_t bytes{};
};

using KernelViewLayout = std::vector<KernelViewSlot>;

[[nodiscard]] inline const KernelViewSlot *
FindKernelViewSlot(const KernelViewLayout &layout, const std::uint64_t binding,
                   const std::uint64_t bytes) noexcept {
  for (const KernelViewSlot &entry : layout) {
    if (entry.binding == binding && entry.bytes == bytes) {
      return &entry;
    }
  }
  return nullptr;
}

} // namespace rund::node::accel::detail
