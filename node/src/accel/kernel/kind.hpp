#pragma once

#include <kernel/program/compute/graph/schema.hpp>

#include <array>
#include <cstddef>

namespace rund::node::accel::detail {

inline constexpr std::size_t kGraphKindSlotCount =
    static_cast<std::size_t>(rund::kernel::NodeKind::ScatterReduce) + 1u;

template <typename T> using GraphKindTable = std::array<T, kGraphKindSlotCount>;

[[nodiscard]] inline constexpr std::size_t
GraphKindSlot(const rund::kernel::NodeKind kind) noexcept {
  return static_cast<std::size_t>(kind);
}

} // namespace rund::node::accel::detail
