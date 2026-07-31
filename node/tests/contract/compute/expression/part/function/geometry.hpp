#pragma once

#include "../../support/record.hpp"

namespace rund::node::test_contract::expression {
namespace {

template <class T, class Executor>
  requires is_fixed<T>
[[nodiscard]] int check_geometry(Executor &executor,
                                 const rund::compute::Backend backend,
                                 const std::array<T, 3u> &input) {
  using namespace rund::compute;
  if (const int result = check_record_parity(
          executor, backend, input, "public-fixed-metric-helper-arities",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto q = fixed(FixedOp::Quarter, value);
            const auto h = fixed(FixedOp::Half, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(len(MetricOp::Squared, value, q))),
                field<Field<1>>(store(len(MetricOp::Squared, value, q, h))),
                field<Field<2>>(store(len(value, q))),
                field<Field<3>>(store(len(value, q, h))),
                field<Field<4>>(store(len(Norm::L1, value, q))),
                field<Field<5>>(store(len(Norm::L1, value, q, h))),
                field<Field<6>>(store(len(Norm::LInf, value, q))),
                field<Field<7>>(store(len(Norm::LInf, value, q, h))),
                field<Field<8>>(store(dist(MetricOp::Squared, value, q, h, q))),
                field<Field<9>>(
                    store(dist(MetricOp::Squared, value, q, h, h, q, value))),
                field<Field<10>>(store(dist(value, q, h, q))),
                field<Field<11>>(store(dist(value, q, h, h, q, value))),
                field<Field<12>>(store(dist(Norm::L1, value, q, h, q))),
                field<Field<13>>(
                    store(dist(Norm::L1, value, q, h, h, q, value))),
                field<Field<14>>(store(dist(Norm::LInf, value, q, h, q))),
                field<Field<15>>(
                    store(dist(Norm::LInf, value, q, h, h, q, value))));
          });
      result != 0) {
    return result;
  }
  if (const int result = check_record_parity(
          executor, backend, input, "public-fixed-metric-helper-angle",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto q = fixed(FixedOp::Quarter, value);
            const auto h = fixed(FixedOp::Half, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(angle(AngleOp::Cosine, value, q, h, q))),
                field<Field<1>>(
                    store(angle(AngleOp::Cosine, value, q, h, h, q, value))));
          });
      result != 0) {
    return 20 + result;
  }
  if (const int result = check_record_parity(
          executor, backend, input,
          "public-fixed-geometry-helper-proportion-cross",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto zero = fixed_zero(value);
            const auto q = fixed(FixedOp::Quarter, value);
            const auto h = fixed(FixedOp::Half, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(proportion(value, h))),
                field<Field<1>>(store(proportion(Axis::X, value, h))),
                field<Field<2>>(store(proportion(Axis::Y, value, h))),
                field<Field<3>>(store(proportion(Axis::X, value, q, h))),
                field<Field<4>>(store(proportion(Axis::Y, value, q, h))),
                field<Field<5>>(store(proportion(Axis::Z, value, q, h))),
                field<Field<6>>(store(cross(value, q, h, q))),
                field<Field<7>>(
                    store(cross(Axis::X, value, q, h, h, q, value))),
                field<Field<8>>(
                    store(cross(Axis::Y, value, q, h, h, q, value))),
                field<Field<9>>(
                    store(cross(Axis::Z, value, q, h, h, q, value))),
                field<Field<10>>(store(orient(zero, zero, h, zero, value, q))),
                field<Field<11>>(store(
                    bary(Axis::X, value, q, zero, zero, h, zero, zero, h))),
                field<Field<12>>(store(
                    bary(Axis::Y, value, q, zero, zero, h, zero, zero, h))),
                field<Field<13>>(store(
                    bary(Axis::Z, value, q, zero, zero, h, zero, zero, h))));
          });
      result != 0) {
    return 40 + result;
  }
  if (const int result = check_record_parity(
          executor, backend, input, "public-fixed-geometry-helper-vector",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto q = fixed(FixedOp::Quarter, value);
            const auto h = fixed(FixedOp::Half, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(unit(Axis::X, value, h))),
                field<Field<1>>(store(unit(Axis::Y, value, h))),
                field<Field<2>>(store(unit(Axis::X, value, q, h))),
                field<Field<3>>(store(unit(Axis::Y, value, q, h))),
                field<Field<4>>(store(unit(Axis::Z, value, q, h))),
                field<Field<5>>(store(proj(Axis::X, value, q, h, q))),
                field<Field<6>>(store(proj(Axis::Y, value, q, h, q))),
                field<Field<7>>(store(proj(Axis::X, value, q, h, h, q, value))),
                field<Field<8>>(store(proj(Axis::Y, value, q, h, h, q, value))),
                field<Field<9>>(
                    store(proj(Axis::Z, value, q, h, h, q, value))));
          });
      result != 0) {
    return 60 + result;
  }
  if (const int result = check_record_parity(
          executor, backend, input,
          "public-fixed-geometry-helper-reject-reflect",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto q = fixed(FixedOp::Quarter, value);
            const auto h = fixed(FixedOp::Half, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(reject(Axis::X, value, q, h, q))),
                field<Field<1>>(store(reject(Axis::Y, value, q, h, q))),
                field<Field<2>>(
                    store(reject(Axis::X, value, q, h, h, q, value))),
                field<Field<3>>(
                    store(reject(Axis::Y, value, q, h, h, q, value))),
                field<Field<4>>(
                    store(reject(Axis::Z, value, q, h, h, q, value))),
                field<Field<5>>(store(reflect(Axis::X, value, q, h, q))),
                field<Field<6>>(store(reflect(Axis::Y, value, q, h, q))),
                field<Field<7>>(
                    store(reflect(Axis::X, value, q, h, h, q, value))),
                field<Field<8>>(
                    store(reflect(Axis::Y, value, q, h, h, q, value))),
                field<Field<9>>(
                    store(reflect(Axis::Z, value, q, h, h, q, value))),
                field<Field<10>>(
                    store(triple(value, q, h, h, q, value, q, value, h))));
          });
      result != 0) {
    return 80 + result;
  }
  if (const int result = check_record_parity(
          executor, backend, input, "public-fixed-geometry-helper-line-plane",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto zero = fixed_zero(value);
            const auto q = fixed(FixedOp::Quarter, value);
            const auto h = fixed(FixedOp::Half, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(
                    line(GeometryOp::Parameter, value, q, zero, zero, h, q))),
                field<Field<1>>(
                    store(line(GeometryOp::Distance, MetricOp::Squared, value,
                               q, zero, zero, h, q))),
                field<Field<2>>(store(
                    line(GeometryOp::Distance, value, q, zero, zero, h, q))),
                field<Field<3>>(store(line(GeometryOp::Projection, Axis::X,
                                           value, q, zero, zero, h, q))),
                field<Field<4>>(store(line(GeometryOp::Projection, Axis::Y,
                                           value, q, zero, zero, h, q))),
                field<Field<5>>(store(plane(GeometryOp::Parameter, value, q, h,
                                            zero, zero, zero, q, h, q))),
                field<Field<6>>(
                    store(plane(GeometryOp::Distance, MetricOp::Squared, value,
                                q, h, zero, zero, zero, q, h, q))),
                field<Field<7>>(store(plane(GeometryOp::Distance, value, q, h,
                                            zero, zero, zero, q, h, q))),
                field<Field<8>>(
                    store(plane(GeometryOp::Projection, Axis::X, value, q, h,
                                zero, zero, zero, q, h, q))),
                field<Field<9>>(
                    store(plane(GeometryOp::Projection, Axis::Y, value, q, h,
                                zero, zero, zero, q, h, q))),
                field<Field<10>>(
                    store(plane(GeometryOp::Projection, Axis::Z, value, q, h,
                                zero, zero, zero, q, h, q))));
          });
      result != 0) {
    return 100 + result;
  }
  return 0;
}

} // namespace
} // namespace rund::node::test_contract::expression
