#pragma once

namespace rund::compute_dsl {

struct MeanOpAbs final {};
struct MeanOpSquared final {};

struct MeanOp final {
  inline static constexpr MeanOpAbs Abs{};
  inline static constexpr MeanOpSquared Squared{};
};

} // namespace rund::compute_dsl
