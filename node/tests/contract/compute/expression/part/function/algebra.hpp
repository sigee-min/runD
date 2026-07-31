#pragma once

#include "../../support/record.hpp"

namespace rund::node::test_contract::expression {
namespace {

template <class T, class Executor>
  requires is_fixed<T>
[[nodiscard]] int check_algebra(Executor &executor,
                                const rund::compute::Backend backend,
                                const std::array<T, 3u> &input) {
  using namespace rund::compute;
  if (const int result = check_record_parity(
          executor, backend, input, "public-fixed-algebra-helper-matrix",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto q = fixed(FixedOp::Quarter, value);
            const auto h = fixed(FixedOp::Half, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(mat(MatOp::Determinant, h, q, q, h))),
                field<Field<1>>(
                    store(mat(MatOp::Determinant, h, q, q, q, h, q, q, q, h))),
                field<Field<2>>(store(mat(Axis::X, h, q, value, q))),
                field<Field<3>>(store(mat(Axis::Y, q, h, value, q))),
                field<Field<4>>(store(mat(Axis::X, h, q, q, value, q, h))),
                field<Field<5>>(store(mat(Axis::Y, q, h, q, value, q, h))),
                field<Field<6>>(store(mat(Axis::Z, q, q, h, value, q, h))),
                field<Field<7>>(
                    store(mat(MatOp::Solve, Axis::X, h, q, q, h, value, q))),
                field<Field<8>>(
                    store(mat(MatOp::Solve, Axis::Y, h, q, q, h, value, q))),
                field<Field<9>>(store(mat(MatOp::Trace, value, q))),
                field<Field<10>>(store(mat(MatOp::Trace, value, q, h))),
                field<Field<11>>(
                    store(mat(MatOp::Transpose, Axis::X, h, q, value, q))),
                field<Field<12>>(
                    store(mat(MatOp::Transpose, Axis::Y, q, h, value, q))),
                field<Field<13>>(store(
                    mat(MatOp::Transpose, Axis::X, h, q, q, value, q, h))),
                field<Field<14>>(store(
                    mat(MatOp::Transpose, Axis::Y, q, h, q, value, q, h))),
                field<Field<15>>(store(
                    mat(MatOp::Transpose, Axis::Z, q, q, h, value, q, h))));
          });
      result != 0) {
    return result;
  }
  if (const int result = check_record_parity(
          executor, backend, input,
          "public-fixed-algebra-helper-affine-polynomial",
          [](auto value) {
            using V = typename decltype(value)::Value;
            const auto q = fixed(FixedOp::Quarter, value);
            const auto h = fixed(FixedOp::Half, value);
            const auto store = [](auto expression) {
              return quantize<V, Rounding::NearestEven, Overflow::Saturate,
                              Approximation::Deterministic>(expression);
            };
            return record(
                field<Field<0>>(store(aff(Axis::X, h, q, q, value, h))),
                field<Field<1>>(store(aff(Axis::Y, q, h, q, value, h))),
                field<Field<2>>(store(aff(Axis::X, h, q, q, q, value, q, h))),
                field<Field<3>>(store(aff(Axis::Y, q, h, q, q, value, q, h))),
                field<Field<4>>(store(aff(Axis::Z, q, q, h, q, value, q, h))),
                field<Field<5>>(store(mix(value, q, h, q))),
                field<Field<6>>(store(mix(value, q, h, q, h, q))),
                field<Field<7>>(store(mix(value, q, h, value, q, h, q, q))),
                field<Field<8>>(store(weighted_mean(value, q, h, q))),
                field<Field<9>>(store(weighted_mean(value, q, h, q, h, q))),
                field<Field<10>>(
                    store(weighted_mean(value, q, h, value, q, h, q, q))),
                field<Field<11>>(store(poly(value, q, h, q))),
                field<Field<12>>(store(poly(value, q, h, q, h))),
                field<Field<13>>(store(poly_deriv(value, h, q))),
                field<Field<14>>(store(poly_deriv(value, h, q, h))));
          });
      result != 0) {
    return 20 + result;
  }
  return 0;
}

} // namespace
} // namespace rund::node::test_contract::expression
