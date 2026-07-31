#pragma once

namespace rund::compute_dsl {

struct StandardizedOpCubic final {};
struct StandardizedOpQuartic final {};

struct StandardizedOp final {
  inline static constexpr StandardizedOpCubic Cubic{};
  inline static constexpr StandardizedOpQuartic Quartic{};
};

} // namespace rund::compute_dsl
