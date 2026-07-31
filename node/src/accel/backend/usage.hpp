#pragma once

#include <accel/buffer.hpp>

#include <kernel/program/compute/binding/model.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

[[nodiscard]] constexpr bool
KnownUsage(const rund::BufferUsage usage) noexcept {
  switch (usage) {
  case rund::BufferUsage::ReadOnly:
  case rund::BufferUsage::WriteOnly:
  case rund::BufferUsage::ReadWrite:
    return true;
  }
  return false;
}

[[nodiscard]] constexpr bool
UsageCompatible(const rund::BufferUsage requested,
                const rund::BufferUsage available) noexcept {
  if (!KnownUsage(requested) || !KnownUsage(available)) {
    return false;
  }
  if (requested == rund::BufferUsage::ReadOnly) {
    return available != rund::BufferUsage::WriteOnly;
  }
  if (requested == rund::BufferUsage::WriteOnly) {
    return available != rund::BufferUsage::ReadOnly;
  }
  return available == rund::BufferUsage::ReadWrite;
}

[[nodiscard]] constexpr std::uint32_t
ResidentUsage(const rund::BufferUsage usage) noexcept {
  return usage == rund::BufferUsage::ReadOnly
             ? rund::kernel::kResidentUsageRead
             : rund::kernel::kResidentUsageWrite;
}

} // namespace rund::node::accel::detail
