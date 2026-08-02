#pragma once

#include "../../array.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

#include <kernel/program/compute/binding/model.hpp>

namespace rund::node::accel::detail {

struct KernelViewSlot final {
  std::uint64_t binding{};
  std::size_t slot{std::numeric_limits<std::size_t>::max()};
  std::uint64_t backing_bytes{};
  std::uint64_t offset_bytes{};
  std::uint64_t count{};
  std::uint64_t stride_bytes{};
  std::uint64_t element_bytes{};
  std::uint32_t usage{};
};

using KernelViewLayout =
    ::rund::node::detail::PreparedArray<KernelViewSlot>;

[[nodiscard]] inline const KernelViewSlot *
FindKernelViewSlot(const KernelViewLayout &layout, const std::uint64_t binding,
                   const rund::kernel::ResidentBufferRef &ref) noexcept {
  for (const KernelViewSlot &entry : layout) {
    if (entry.binding == binding && entry.backing_bytes == ref.bytes &&
        entry.offset_bytes == ref.offset_bytes && entry.count == ref.count &&
        entry.stride_bytes == ref.stride_bytes &&
        entry.element_bytes == ref.element_bytes && entry.usage == ref.usage) {
      return &entry;
    }
  }
  return nullptr;
}

} // namespace rund::node::accel::detail
