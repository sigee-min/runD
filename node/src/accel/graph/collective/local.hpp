#pragma once

#include <accel/buffer.hpp>

#include <node/accel/context.hpp>

#include <kernel/program/compute/sort/model.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

[[nodiscard]] inline bool CollectiveUsageOk(const rund::BufferUsage usage,
                                            const bool write) noexcept {
  return usage == rund::BufferUsage::ReadWrite ||
         usage == (write ? rund::BufferUsage::WriteOnly
                         : rund::BufferUsage::ReadOnly);
}

[[nodiscard]] inline bool SortKeyWidthOk(const std::uint64_t width) noexcept {
  return width == sizeof(rund::kernel::u32) ||
         width == sizeof(rund::kernel::u64);
}

[[nodiscard]] inline rund::kernel::SortKey
SortKeyForWidth(const std::uint64_t width) noexcept {
  return width == sizeof(rund::kernel::u64) ? rund::kernel::SortKey::U64
                                            : rund::kernel::SortKey::U32;
}

} // namespace rund::node::accel::detail
