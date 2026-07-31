#pragma once

namespace rund::compute_dsl {

struct SumOpAbs final {};
struct SumOpSquared final {};

struct SumOp final {
  inline static constexpr SumOpAbs Abs{};
  inline static constexpr SumOpSquared Squared{};
};

} // namespace rund::compute_dsl
