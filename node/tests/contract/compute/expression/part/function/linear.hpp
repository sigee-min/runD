#pragma once

#include "../../support/record.hpp"

namespace rund::node::test_contract::expression {
namespace {

template <class T, class Executor>
  requires is_fixed<T>
[[nodiscard]] int check_linear(Executor &executor,
                               const rund::compute::Backend backend,
                               const std::array<T, 3u> &input) {
  using namespace rund::compute;
  if (const int result = check_record_parity(
          executor, backend, input, "public-fixed-linear-helper-arities",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto q = fixed(FixedOp::Quarter, value);
            const auto h = fixed(FixedOp::Half, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(dot(value, h, q, value))),
                field<Field<1>>(store(dot(value, h, q, q, value, h))),
                field<Field<2>>(store(dot(value, h, q, value, q, value, h, q))),
                field<Field<3>>(
                    store(dot(value, h, q, value, h, q, value, h, q, value))),
                field<Field<4>>(store(
                    dot(value, h, q, value, h, q, q, value, h, q, value, h))),
                field<Field<5>>(store(dot(value, h, q, value, h, q, value, h, q,
                                          value, h, q, value, h, q, value))),
                field<Field<6>>(store(conv(value, h, q, q, h, value))),
                field<Field<7>>(
                    store(conv(value, h, q, value, h, q, value, h, q, value))),
                field<Field<8>>(store(conv(value, h, q, value, h, q, value, q,
                                           value, h, q, value, h, q))),
                field<Field<9>>(
                    store(conv(value, h, q, value, h, q, value, h, q, q, value,
                               h, q, value, h, q, value, h))));
          });
      result != 0) {
    return result;
  }
  if (const int result = check_record_parity(
          executor, backend, input, "public-fixed-stats-helper-means",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto q = fixed(FixedOp::Quarter, value);
            const auto h = fixed(FixedOp::Half, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(mean(value, q))),
                field<Field<1>>(store(mean(value, q, h))),
                field<Field<2>>(store(mean(value, q, h, value))),
                field<Field<3>>(store(mean(MeanOp::Abs, value, q))),
                field<Field<4>>(store(mean(MeanOp::Abs, value, q, h))),
                field<Field<5>>(store(mean(MeanOp::Abs, value, q, h, value))),
                field<Field<6>>(store(mean(MeanOp::Squared, value, q))),
                field<Field<7>>(store(mean(MeanOp::Squared, value, q, h))),
                field<Field<8>>(
                    store(mean(MeanOp::Squared, value, q, h, value))),
                field<Field<9>>(store(centered(value, h))),
                field<Field<10>>(store(centered(CenteredOp::Abs, value, h))),
                field<Field<11>>(
                    store(centered(CenteredOp::Squared, value, h))),
                field<Field<12>>(store(centered(CenteredOp::Cubic, value, h))),
                field<Field<13>>(
                    store(centered(CenteredOp::Quartic, value, h))));
          });
      result != 0) {
    return 20 + result;
  }
  if (const int result = check_record_parity(
          executor, backend, input, "public-fixed-stats-helper-sums",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto q = fixed(FixedOp::Quarter, value);
            const auto h = fixed(FixedOp::Half, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(sum(value, q))),
                field<Field<1>>(store(sum(value, q, h))),
                field<Field<2>>(store(sum(value, q, h, value))),
                field<Field<3>>(store(sum(SumOp::Abs, value, q))),
                field<Field<4>>(store(sum(SumOp::Abs, value, q, h))),
                field<Field<5>>(store(sum(SumOp::Abs, value, q, h, value))),
                field<Field<6>>(store(sum(SumOp::Squared, value, q))),
                field<Field<7>>(store(sum(SumOp::Squared, value, q, h))),
                field<Field<8>>(store(sum(SumOp::Squared, value, q, h, value))),
                field<Field<9>>(store(diff(value, h))),
                field<Field<10>>(store(diff(value, q, h))),
                field<Field<11>>(
                    store(diff(DifferenceOrder::Second, value, q, h))),
                field<Field<12>>(
                    store(diff(DifferenceOrder::Third, value, q, h, value))),
                field<Field<13>>(store(var(value, q))),
                field<Field<14>>(store(var(value, q, h))),
                field<Field<15>>(store(var(value, q, h, value))));
          });
      result != 0) {
    return 40 + result;
  }
  if (const int result = check_record_parity(
          executor, backend, input, "public-fixed-stats-helper-moments",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto q = fixed(FixedOp::Quarter, value);
            const auto h = fixed(FixedOp::Half, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(rms(value, q))),
                field<Field<1>>(store(rms(value, q, h))),
                field<Field<2>>(store(rms(value, q, h, value))),
                field<Field<3>>(store(ratio(value, h))),
                field<Field<4>>(store(zscore(value, q, h))),
                field<Field<5>>(
                    store(standardized(StandardizedOp::Cubic, value, q, h))),
                field<Field<6>>(
                    store(standardized(StandardizedOp::Quartic, value, q, h))),
                field<Field<7>>(store(mean(StandardizedOp::Cubic, value, q))),
                field<Field<8>>(
                    store(mean(StandardizedOp::Cubic, value, q, h))),
                field<Field<9>>(
                    store(mean(StandardizedOp::Cubic, value, q, h, value))),
                field<Field<10>>(
                    store(mean(StandardizedOp::Quartic, value, q))),
                field<Field<11>>(
                    store(mean(StandardizedOp::Quartic, value, q, h))),
                field<Field<12>>(
                    store(mean(StandardizedOp::Quartic, value, q, h, value))),
                field<Field<13>>(store(cov(value, q, h, value))),
                field<Field<14>>(store(cov(value, q, h, q, h, value))),
                field<Field<15>>(
                    store(cov(value, q, h, value, q, h, value, q))));
          });
      result != 0) {
    return 60 + result;
  }
  if (const int result = check_record_parity(
          executor, backend, input, "public-fixed-stats-helper-correlation",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto q = fixed(FixedOp::Quarter, value);
            const auto h = fixed(FixedOp::Half, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(corr(value, q, h, value))),
                field<Field<1>>(store(corr(value, q, h, q, h, value))),
                field<Field<2>>(
                    store(corr(value, q, h, value, q, h, value, q))),
                field<Field<3>>(
                    store(standard_mean(StandardizedOp::Cubic, value, q))),
                field<Field<4>>(
                    store(standard_mean(StandardizedOp::Cubic, value, q, h))),
                field<Field<5>>(store(
                    standard_mean(StandardizedOp::Cubic, value, q, h, value))),
                field<Field<6>>(
                    store(standard_mean(StandardizedOp::Quartic, value, q))),
                field<Field<7>>(
                    store(standard_mean(StandardizedOp::Quartic, value, q, h))),
                field<Field<8>>(store(standard_mean(StandardizedOp::Quartic,
                                                    value, q, h, value))));
          });
      result != 0) {
    return 80 + result;
  }
  return 0;
}

} // namespace
} // namespace rund::node::test_contract::expression
