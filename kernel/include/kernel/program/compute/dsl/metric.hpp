#pragma once

namespace rund::compute_dsl {

struct MetricOpSquared final {};
struct AngleOpCosine final {};

struct MetricOp final {
  inline static constexpr MetricOpSquared Squared{};
};

struct AngleOp final {
  inline static constexpr AngleOpCosine Cosine{};
};

} // namespace rund::compute_dsl
