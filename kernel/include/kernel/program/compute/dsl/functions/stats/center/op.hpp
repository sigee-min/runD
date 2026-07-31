#pragma once

namespace rund::compute_dsl {

struct CenteredOpAbs final {};
struct CenteredOpSquared final {};
struct CenteredOpCubic final {};
struct CenteredOpQuartic final {};

struct CenteredOp final {
  inline static constexpr CenteredOpAbs Abs{};
  inline static constexpr CenteredOpSquared Squared{};
  inline static constexpr CenteredOpCubic Cubic{};
  inline static constexpr CenteredOpQuartic Quartic{};
};

} // namespace rund::compute_dsl
