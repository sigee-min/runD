#pragma once

namespace rund::compute_dsl {

struct FixedOpHalf final {};
struct FixedOpThird final {};
struct FixedOpQuarter final {};

struct FixedOp final {
  inline static constexpr FixedOpHalf Half{};
  inline static constexpr FixedOpThird Third{};
  inline static constexpr FixedOpQuarter Quarter{};
};

} // namespace rund::compute_dsl
