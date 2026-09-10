#pragma once

#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {
inline constexpr std::uint32_t NoNode =
    std::numeric_limits<std::uint32_t>::max();
} // namespace rund::node::accel::detail
