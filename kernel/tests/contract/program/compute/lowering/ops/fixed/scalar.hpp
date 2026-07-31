#pragma once

#include "../base.hpp"

namespace program_compute_contract::lowering_support {

[[nodiscard]] inline auto BuildFixedLane32ScalarOps() {
  i32 lhs[4]{};
  i32 rhs[4]{};
  i32 out[4]{};
  const auto body = rund::compute_dsl::bind(4u)
                        .fixed<1, 31>()
                        .param<"marker">(7)
                        .read<"lhs">(lhs)
                        .read<"rhs">(rhs)
                        .write<"out">(out);

  return rund::compute_dsl::def("lower-fixed-scalar-lane32")
      .on(body)
      .map([](auto i, auto b) {
        auto marker = b.template param<"marker">();
        auto lhs = b.template read<"lhs">();
        auto rhs = b.template read<"rhs">();
        auto out = b.template write<"out">();

        auto sum = lhs[i] + 3;
        auto diff = 5 - rhs[i];
        auto product = sum * 2;
        auto comparison = rund::compute_dsl::predicate_and(
            rund::compute_dsl::ne(lhs[i], marker),
            rund::compute_dsl::predicate_or(
                rund::compute_dsl::gt(product, diff),
                rund::compute_dsl::ge(rhs[i], 0)));
        auto unary = rund::compute_dsl::neg(lhs[i]) + (-rhs[i]) +
                     rund::compute_dsl::abs(lhs[i]) +
                     rund::compute_dsl::abs_magnitude(rhs[i]) +
                     rund::compute_dsl::sign(rhs[i]);
        out[i] = rund::compute_dsl::select(
            rund::compute_dsl::predicate_not(comparison), unary, 0);
      });
}

[[nodiscard]] inline auto BuildFixedLane64ScalarOps() {
  i64 lhs[4]{};
  i64 rhs[4]{};
  i64 out[4]{};
  const auto body = rund::compute_dsl::bind(4u)
                        .fixed<1, 63>()
                        .param<"marker">(i64{7})
                        .read<"lhs">(lhs)
                        .read<"rhs">(rhs)
                        .write<"out">(out);

  return rund::compute_dsl::def("lower-fixed-scalar-lane64")
      .on(body)
      .map([](auto i, auto b) {
        auto marker = b.template param<"marker">();
        auto lhs = b.template read<"lhs">();
        auto rhs = b.template read<"rhs">();
        auto out = b.template write<"out">();

        auto sum = lhs[i] + i64{3};
        auto diff = i64{5} - rhs[i];
        auto product = rund::compute_dsl::quantize<1, 63>(sum) * i64{2};
        auto comparison = rund::compute_dsl::predicate_and(
            rund::compute_dsl::ne(lhs[i], marker),
            rund::compute_dsl::predicate_or(
                rund::compute_dsl::gt(product, diff),
                rund::compute_dsl::ge(rhs[i], i64{0})));
        auto unary = rund::compute_dsl::neg(lhs[i]) + (-rhs[i]) +
                     rund::compute_dsl::abs(lhs[i]) +
                     rund::compute_dsl::abs_magnitude(rhs[i]) +
                     rund::compute_dsl::sign(rhs[i]);
        out[i] = rund::compute_dsl::select(
            rund::compute_dsl::predicate_not(comparison), unary, i64{0});
      });
}

} // namespace program_compute_contract::lowering_support
