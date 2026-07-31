#pragma once

#include "../support/record.hpp"

namespace rund::node::test_contract::expression {
namespace {

template <class Executor>
[[nodiscard]] int check_left_rescale(Executor &executor,
                                     const rund::compute::Backend backend) {
  using namespace rund::compute;
  using Source = Fixed<63, 1>;
  using Target = Fixed<1, 63>;
  using SourceRaw = typename Source::Raw;
  using TargetRaw = typename Target::Raw;
  constexpr SourceRaw minimum = std::numeric_limits<SourceRaw>::min();
  constexpr SourceRaw maximum = std::numeric_limits<SourceRaw>::max();
  constexpr TargetRaw scaled_one = TargetRaw{1} << 61u;
  const std::array<Source, 4u> input{Source::from_raw(minimum),
                                     Source::from_raw(maximum),
                                     Source::from_raw(1), Source::from_raw(-1)};
  const std::array<std::array<Target, 4u>, 2u> expected{
      std::array<Target, 4u>{Target::max(), Target::max(),
                             Target::from_raw(scaled_one),
                             Target::from_raw(scaled_one)},
      std::array<Target, 4u>{Target::zero(), Target::from_raw(scaled_one),
                             Target::from_raw(scaled_one),
                             Target::from_raw(scaled_one)},
  };
  return check_record_expected(
      executor, backend, input, expected,
      "public-fixed-left-rescale-extrema-golden", [](auto value) {
        const auto squared = value * value;
        return record(
            field<Field<0>>(
                quantize<Target, Rounding::NearestEven, Overflow::Saturate,
                         Approximation::Exact>(squared)),
            field<Field<1>>(
                quantize<Target, Rounding::NearestEven, Overflow::Wrap,
                         Approximation::Exact>(squared)));
      });
}

template <class T, class Executor>
  requires is_fixed<T>
[[nodiscard]] int check_policy_literals(Executor &executor,
                                        const rund::compute::Backend backend) {
  using namespace rund::compute;
  using Raw = typename T::Raw;
  constexpr Raw minimum = std::numeric_limits<Raw>::min();
  constexpr Raw maximum = std::numeric_limits<Raw>::max();
  constexpr Raw half = Raw{1} << (T::fraction_bits - 1u);
  const std::array<T, 5u> input{T::from_raw(minimum), T::from_raw(maximum),
                                T::zero(), T::from_raw(1), T::from_raw(-1)};
  const std::array<std::array<T, 5u>, 10u> expected{
      std::array<T, 5u>{T::from_raw(1), T::from_raw(-1), T::from_raw(1),
                        T::from_raw(3), T::from_raw(-1)},
      std::array<T, 5u>{T::from_raw(-1), T::from_raw(-1), T::from_raw(-1),
                        T::from_raw(2), T::from_raw(-1)},
      std::array<T, 5u>{T::zero(), T::from_raw(-2), T::zero(), T::from_raw(2),
                        T::from_raw(-2)},
      std::array<T, 5u>{T::zero(), T::from_raw(-2), T::zero(), T::from_raw(1),
                        T::from_raw(-2)},
      std::array<T, 5u>{T::zero(), T::from_raw(-1), T::zero(), T::from_raw(2),
                        T::from_raw(-1)},
      std::array<T, 5u>{T::from_raw(1), T::from_raw(1), T::from_raw(1),
                        T::from_raw(1), T::from_raw(1)},
      std::array<T, 5u>{T::from_raw(half), T::from_raw(half), T::from_raw(half),
                        T::from_raw(half), T::from_raw(half)},
      std::array<T, 5u>{T::zero(), T::from_raw(-2), T::zero(), T::from_raw(2),
                        T::from_raw(-2)},
      std::array<T, 5u>{T::zero(), T::zero(), T::zero(), T::zero(), T::zero()},
      std::array<T, 5u>{T::zero(), T::from_raw(2), T::zero(), T::from_raw(2),
                        T::from_raw(2)},
  };
  return check_record_expected(
      executor, backend, input, expected,
      "public-fixed-custom-policy-literal-helper-golden", [](auto value) {
        using V = typename decltype(value)::Value;
        const auto q =
            quantize<V, Rounding::Down, Overflow::Wrap, Approximation::Exact>(
                value + value);
        const auto store = [](auto expression) {
          return quantize<V, Rounding::Down, Overflow::Wrap,
                          Approximation::Exact>(expression);
        };
        const auto zero = fixed_zero(q);
        const auto half_value = fixed(FixedOp::Half, q);
        const auto quarter = fixed(FixedOp::Quarter, q);
        return record(field<Field<0>>(store(q + V::from_raw(1))),
                      field<Field<1>>(store(
                          select(q > V::from_raw(0), q, V::from_raw(-1)))),
                      field<Field<2>>(store(clamp(q, V::min(), V::max()))),
                      field<Field<3>>(store(min(q, V::from_raw(1)))),
                      field<Field<4>>(store(max(q, V::from_raw(-1)))),
                      field<Field<5>>(store(
                          select(q == q, V::from_raw(1), V::from_raw(1)))),
                      field<Field<6>>(store(zero + half_value)),
                      field<Field<7>>(store(clip(q, half_value))),
                      field<Field<8>>(store(huber(q, quarter))),
                      field<Field<9>>(store(len(Norm::L1, q, zero))));
      });
}

template <class T, class Executor>
  requires is_fixed<T>
[[nodiscard]] int check_widened_ternary(Executor &executor,
                                        const rund::compute::Backend backend) {
  using namespace rund::compute;
  using Raw = typename T::Raw;
  constexpr Raw minimum = std::numeric_limits<Raw>::min();
  constexpr Raw maximum = std::numeric_limits<Raw>::max();
  const std::array<T, 5u> input{T::from_raw(minimum), T::from_raw(maximum),
                                T::zero(), T::from_raw(1), T::from_raw(-1)};
  const std::array<std::array<T, 5u>, 5u> expected{
      std::array<T, 5u>{T::zero(), T::from_raw(maximum), T::zero(),
                        T::from_raw(1), T::from_raw(-2)},
      std::array<T, 5u>{T::from_raw(minimum), T::from_raw(-2), T::zero(),
                        T::from_raw(2), T::from_raw(-1)},
      std::array<T, 5u>{T::zero(), T::from_raw(maximum), T::zero(),
                        T::from_raw(1), T::zero()},
      std::array<T, 5u>{T::zero(), T::from_raw(maximum), T::zero(),
                        T::from_raw(2), T::zero()},
      input,
  };
  return check_record_expected(
      executor, backend, input, expected,
      "public-fixed-widened-ternary-cancellation-golden", [](auto value) {
        using V = typename decltype(value)::Value;
        const auto sum = value + value;
        const auto difference = value - value;
        const auto store_wrap = [](auto expression) {
          return quantize<V, Rounding::NearestEven, Overflow::Wrap,
                          Approximation::Exact>(expression);
        };
        const auto store_saturate = [](auto expression) {
          return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                          Approximation::Exact>(expression);
        };
        return record(field<Field<0>>(store_wrap(min(sum, value))),
                      field<Field<1>>(store_wrap(max(sum, value))),
                      field<Field<2>>(store_saturate(
                          clamp(sum, difference, abs_magnitude(value)))),
                      field<Field<3>>(store_saturate(
                          select(value >= V::from_raw(0), sum, difference))),
                      field<Field<4>>(store_saturate(sum - value)));
      });
}

} // namespace
} // namespace rund::node::test_contract::expression
