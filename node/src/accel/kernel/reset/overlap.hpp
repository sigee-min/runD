#pragma once

#include "../backend/run.hpp"

#include <span>

namespace rund::node::accel::detail::reset {

[[nodiscard]] bool Compatible(std::span<const BoundReset> resets) noexcept;

} // namespace rund::node::accel::detail::reset
