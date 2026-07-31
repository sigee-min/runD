#pragma once

#include "../../support/record.hpp"

namespace rund::node::test_contract::expression {
namespace {

template <class T, class Executor>
  requires is_fixed<T>
[[nodiscard]] int check_unit_golden(Executor &executor,
                                    const rund::compute::Backend backend) {
  using namespace rund::compute;
  using Raw = typename T::Raw;
  constexpr Raw one_raw = Raw{1} << T::fraction_bits;
  constexpr Raw half_raw = one_raw >> 1u;
  constexpr Raw quarter_raw = one_raw >> 2u;
  constexpr Raw three_quarter_raw = half_raw + quarter_raw;
  const T zero = T::zero();
  const T one = T::from_raw(one_raw);
  const T half = T::from_raw(half_raw);
  const T quarter = T::from_raw(quarter_raw);
  const T third = fixed_nearest<T>(1u, 3u);
  const T two = T::from_raw(one_raw * 2);
  const std::array<T, 5u> input{T::from_raw(-one_raw), zero, half, one, two};
  const std::array<std::array<T, 5u>, 13u> expected{
      std::array<T, 5u>{one, one, one, one, one},
      std::array<T, 5u>{T::max(), T::max(), T::max(), T::max(), T::max()},
      std::array<T, 5u>{half, half, half, half, half},
      std::array<T, 5u>{third, third, third, third, third},
      std::array<T, 5u>{quarter, quarter, quarter, quarter, quarter},
      std::array<T, 5u>{zero, zero, half, one, one},
      std::array<T, 5u>{zero, zero, one, one, one},
      std::array<T, 5u>{zero, zero, half, one, one},
      std::array<T, 5u>{zero, zero, half, one, one},
      std::array<T, 5u>{zero, half, T::from_raw(three_quarter_raw), one, one},
      std::array<T, 5u>{T::from_raw(-one_raw), zero, half, one, one},
      std::array<T, 5u>{zero, zero, one, zero, zero},
      std::array<T, 5u>{zero, zero, one, zero, zero},
  };
  return check_record_expected(
      executor, backend, input, expected, "fixed-format-aware-unit-golden",
      [](auto value) {
        using V = typename decltype(value)::Value;
        const auto zero = fixed_zero(value);
        const auto half = fixed(FixedOp::Half, value);
        const auto one = fixed_one(value);
        const auto store = [](auto expression) {
          return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                          Approximation::Exact>(expression);
        };
        return record(
            field<Field<0>>(store(one)),
            field<Field<1>>(store(fixed_max(value))),
            field<Field<2>>(store(half)),
            field<Field<3>>(store(fixed(FixedOp::Third, value))),
            field<Field<4>>(store(fixed(FixedOp::Quarter, value))),
            field<Field<5>>(store(saturate(value))),
            field<Field<6>>(store(step(half, value))),
            field<Field<7>>(store(lerp(zero, one, value))),
            field<Field<8>>(store(fade(value))),
            field<Field<9>>(
                store(activation(ActivationOp::HardSigmoid, value))),
            field<Field<10>>(store(activation(ActivationOp::HardTanh, value))),
            field<Field<11>>(store(window(WindowOp::Parabolic, value))),
            field<Field<12>>(store(window(WindowOp::Triangular, value))));
      });
}

template <class T, class Executor>
  requires is_fixed<T>
[[nodiscard]] int check_window_golden(Executor &executor,
                                      const rund::compute::Backend backend) {
  using namespace rund::compute;
  using Raw = typename T::Raw;
  constexpr Raw one_raw = Raw{1} << T::fraction_bits;
  constexpr Raw half_raw = one_raw >> 1u;
  const T zero = T::zero();
  const T half = T::from_raw(half_raw);
  const T one = T::from_raw(one_raw);
  const T hamming_a = fixed_q31<T>(0x451eb852u);
  const T hamming_b = fixed_q31<T>(0x3ae147aeu);
  const T blackman_a0 = fixed_q31<T>(0x35c28f5cu);
  const T blackman_a2 = fixed_q31<T>(0x0a3d70a3u);
  const T hamming_edge =
      T::from_raw(static_cast<Raw>(hamming_a.raw() - hamming_b.raw()));
  const T hamming_center =
      T::from_raw(static_cast<Raw>(hamming_a.raw() + hamming_b.raw()));
  const T blackman_edge = T::from_raw(
      static_cast<Raw>(blackman_a0.raw() - half.raw() + blackman_a2.raw()));
  const T blackman_center = T::from_raw(
      static_cast<Raw>(blackman_a0.raw() + half.raw() + blackman_a2.raw()));
  const std::array<T, 3u> input{zero, half, one};
  const std::array<std::array<T, 3u>, 5u> expected{
      std::array<T, 3u>{zero, one, zero},
      std::array<T, 3u>{hamming_edge, hamming_center, hamming_edge},
      std::array<T, 3u>{blackman_edge, blackman_center, blackman_edge},
      std::array<T, 3u>{fixed_unit_hash(zero), fixed_unit_hash(half),
                        fixed_unit_hash(one)},
      std::array<T, 3u>{fixed_unit_hash(zero, half),
                        fixed_unit_hash(half, half),
                        fixed_unit_hash(one, half)},
  };
  return check_record_expected(
      executor, backend, input, expected,
      "fixed-format-aware-window-hash-golden", [](auto value) {
        using V = typename decltype(value)::Value;
        const auto half = fixed(FixedOp::Half, value);
        const auto store = [](auto expression) {
          return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                          Approximation::Deterministic>(expression);
        };
        return record(field<Field<0>>(store(window(WindowOp::Hann, value))),
                      field<Field<1>>(store(window(WindowOp::Hamming, value))),
                      field<Field<2>>(store(window(WindowOp::Blackman, value))),
                      field<Field<3>>(store(hash(HashOp::Unit, value))),
                      field<Field<4>>(store(hash(HashOp::Unit, value, half))));
      });
}

} // namespace
} // namespace rund::node::test_contract::expression
