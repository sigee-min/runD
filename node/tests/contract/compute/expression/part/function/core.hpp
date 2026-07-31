#pragma once

#include "../../support/record.hpp"

namespace rund::node::test_contract::expression {
namespace {

template <class T, class Executor>
  requires is_fixed<T>
[[nodiscard]] int check_core(Executor &executor,
                             const rund::compute::Backend backend,
                             const std::array<T, 3u> &input) {
  using namespace rund::compute;
  if (const int result = check_record_parity(
          executor, backend, input, "public-fixed-core-helper-identities",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto zero = fixed_zero(value);
            const auto half = fixed(FixedOp::Half, value);
            const auto quarter = fixed(FixedOp::Quarter, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(zero)),
                field<Field<1>>(store(fixed_max(value))),
                field<Field<2>>(store(half)),
                field<Field<3>>(store(fixed(FixedOp::Third, value))),
                field<Field<4>>(store(quarter)),
                field<Field<5>>(store(mul_fixed(value, half))),
                field<Field<6>>(store(div_fixed(value, half))),
                field<Field<7>>(store(neg(value))),
                field<Field<8>>(store(select(eq(value, half), half, zero))),
                field<Field<9>>(store(select(ne(value, half), half, zero))),
                field<Field<10>>(store(select(lt(value, half), half, zero))),
                field<Field<11>>(store(select(le(value, half), half, zero))),
                field<Field<12>>(store(select(gt(value, half), half, zero))),
                field<Field<13>>(store(select(ge(value, half), half, zero))),
                field<Field<14>>(
                    store(select(predicate_not(eq(value, half)), half, zero))),
                field<Field<15>>(store(
                    select(predicate_and(ge(value, zero), le(value, half)),
                           half, zero))));
          });
      result != 0) {
    return result;
  }
  if (const int result = check_record_parity(
          executor, backend, input, "public-fixed-core-helper-ranges",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto zero = fixed_zero(value);
            const auto quarter = fixed(FixedOp::Quarter, value);
            const auto half = fixed(FixedOp::Half, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(select(
                    predicate_and(value >= zero, value <= half), half, zero))),
                field<Field<1>>(store(
                    select(predicate_or(value<zero, value> half), half, zero))),
                field<Field<2>>(store(bit_and(value, half))),
                field<Field<3>>(store(bit_or(value, quarter))),
                field<Field<4>>(store(bit_xor(value, half))),
                field<Field<5>>(store(saturate(value))),
                field<Field<6>>(store(step(quarter, value))),
                field<Field<7>>(store(absdiff(value, half))),
                field<Field<8>>(store(midrange(value, half))),
                field<Field<9>>(store(median(value, quarter, half))),
                field<Field<10>>(store(spread(value, quarter, half))),
                field<Field<11>>(store(clamp_range(value, half, quarter))),
                field<Field<12>>(
                    store(select(in_range(value, quarter, half), half, zero))),
                field<Field<13>>(
                    store(select(out_range(value, quarter, half), half, zero))),
                field<Field<14>>(store(bandpass(value, quarter, half))),
                field<Field<15>>(store(bandstop(value, quarter, half))));
          });
      result != 0) {
    return 20 + result;
  }
  if (const int result = check_record_parity(
          executor, backend, input, "public-fixed-core-helper-predicates",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto zero = fixed_zero(value);
            const auto quarter = fixed(FixedOp::Quarter, value);
            const auto half = fixed(FixedOp::Half, value);
            const auto yes = fixed_max(value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            const auto a = value >= zero;
            const auto b = value <= half;
            const auto c = value != quarter;
            const auto d = value == value;
            return record(
                field<Field<0>>(store(select(is_zero(value), yes, zero))),
                field<Field<1>>(store(select(nonzero(value), yes, zero))),
                field<Field<2>>(store(select(is_neg(value), yes, zero))),
                field<Field<3>>(store(select(is_pos(value), yes, zero))),
                field<Field<4>>(store(select(is_nonneg(value), yes, zero))),
                field<Field<5>>(store(select(is_nonpos(value), yes, zero))),
                field<Field<6>>(store(select(all(a, b), yes, zero))),
                field<Field<7>>(store(select(all(a, b, c), yes, zero))),
                field<Field<8>>(store(select(all(a, b, c, d), yes, zero))),
                field<Field<9>>(store(select(any(a, b), yes, zero))),
                field<Field<10>>(store(select(any(a, b, c), yes, zero))),
                field<Field<11>>(store(select(any(a, b, c, d), yes, zero))),
                field<Field<12>>(store(keep_if(a, value))),
                field<Field<13>>(store(zero_if(a, value))),
                field<Field<14>>(
                    store(select(near(value, half, quarter), yes, zero))),
                field<Field<15>>(
                    store(select(near(value, quarter), yes, zero))));
          });
      result != 0) {
    return 40 + result;
  }
  if (const int result = check_record_parity(
          executor, backend, input, "public-fixed-core-helper-interpolation",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto zero = fixed_zero(value);
            const auto quarter = fixed(FixedOp::Quarter, value);
            const auto half = fixed(FixedOp::Half, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(deadzone(value, quarter))),
                field<Field<1>>(store(snap(value, half, quarter))),
                field<Field<2>>(store(clip(value, half))),
                field<Field<3>>(store(positive_part(value))),
                field<Field<4>>(store(negative_part(value))),
                field<Field<5>>(store(lerp(value, half, quarter))),
                field<Field<6>>(
                    store(lerp(value, half, quarter, value, quarter, half))),
                field<Field<7>>(
                    store(lerp(value, half, quarter, value, half, quarter,
                               value, half, quarter, half, quarter))),
                field<Field<8>>(store(unlerp(zero, half, value))),
                field<Field<9>>(store(remap(zero, half, quarter, half, value))),
                field<Field<10>>(store(bezier(value, quarter, half, quarter))),
                field<Field<11>>(
                    store(bezier(value, quarter, half, value, quarter))),
                field<Field<12>>(store(fade(value))),
                field<Field<13>>(store(smoothstep(zero, half, value))));
          });
      result != 0) {
    return 60 + result;
  }
  return 0;
}

} // namespace
} // namespace rund::node::test_contract::expression
