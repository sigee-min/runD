#pragma once

#include "../base.hpp"

namespace program_compute_contract::lowering_support {

[[nodiscard]] inline auto BuildFixedLane32BitOps() {
  i32 lhs[4]{};
  i32 rhs[4]{};
  i32 out[4]{};
  const auto body = rund::compute_dsl::bind(4u)
                        .fixed<1, 31>()
                        .read<"lhs">(lhs)
                        .read<"rhs">(rhs)
                        .write<"out">(out);

  return rund::compute_dsl::def("lower-fixed-bit-lane32")
      .on(body)
      .map([](auto i, auto b) {
        auto lhs = b.template read<"lhs">();
        auto rhs = b.template read<"rhs">();
        auto out = b.template write<"out">();

        const auto masked = rund::compute_dsl::bit_and(lhs[i], rhs[i]);
        const auto mixed = rund::compute_dsl::bit_or(
            masked, rund::compute_dsl::bit_xor(
                        rund::compute_dsl::bit_not(lhs[i]),
                        rund::compute_dsl::shl_const<3>(rhs[i])));
        out[i] = mixed + rund::compute_dsl::shr_logical_const<5>(lhs[i]) +
                 rund::compute_dsl::shr_arithmetic_const<7>(rhs[i]);
      });
}

[[nodiscard]] inline auto BuildFixedLane64BitOps() {
  i64 lhs[4]{};
  i64 rhs[4]{};
  i64 out[4]{};
  const auto body = rund::compute_dsl::bind(4u)
                        .fixed<1, 63>()
                        .read<"lhs">(lhs)
                        .read<"rhs">(rhs)
                        .write<"out">(out);

  return rund::compute_dsl::def("lower-fixed-bit-lane64")
      .on(body)
      .map([](auto i, auto b) {
        auto lhs = b.template read<"lhs">();
        auto rhs = b.template read<"rhs">();
        auto out = b.template write<"out">();

        const auto masked = rund::compute_dsl::bit_and(lhs[i], rhs[i]);
        const auto mixed = rund::compute_dsl::bit_or(
            masked, rund::compute_dsl::bit_xor(
                        rund::compute_dsl::bit_not(lhs[i]),
                        rund::compute_dsl::shl_const<11>(rhs[i])));
        out[i] = mixed + rund::compute_dsl::shr_logical_const<13>(lhs[i]) +
                 rund::compute_dsl::shr_arithmetic_const<17>(rhs[i]);
      });
}

} // namespace program_compute_contract::lowering_support
