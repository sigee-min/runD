#pragma once

#include "../base.hpp"

namespace program_compute_contract::lowering_support {

template <unsigned IntegerBits, unsigned FractionBits, class Raw>
[[nodiscard]] inline auto BuildDeclaredMultiplyOps() {
  Raw mode[4]{};
  Raw lhs[4]{};
  Raw rhs[4]{};
  Raw addend[4]{};
  Raw out[4]{};
  const auto body = rund::compute_dsl::bind(4u)
                        .template fixed<IntegerBits, FractionBits>()
                        .template read<"mode">(mode)
                        .template read<"lhs">(lhs)
                        .template read<"rhs">(rhs)
                        .template read<"addend">(addend)
                        .template write<"out">(out);

  return rund::compute_dsl::def(IntegerBits + FractionBits == 32u
                                    ? "lower-fixed-16-16-declared-multiply"
                                    : "lower-fixed-20-44-declared-multiply")
      .on(body)
      .map([](auto i, auto b) {
        const auto mode = b.template read<"mode">();
        const auto lhs = b.template read<"lhs">();
        const auto rhs = b.template read<"rhs">();
        const auto addend = b.template read<"addend">();
        const auto out = b.template write<"out">();

        auto result =
            rund::compute_dsl::mul_add_fixed(lhs[i], rhs[i], addend[i]);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], Raw{2}),
            rund::compute_dsl::mul_unsigned_fixed(lhs[i], rhs[i]), result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], Raw{1}),
            rund::compute_dsl::mul_fixed_scaled(lhs[i], rhs[i]), result);
        out[i] = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], Raw{0}),
            rund::compute_dsl::mul_fixed(lhs[i], rhs[i]), result);
      });
}

[[nodiscard]] inline auto BuildFixed16_16DeclaredMultiplyOps() {
  return BuildDeclaredMultiplyOps<16u, 16u, i32>();
}

[[nodiscard]] inline auto BuildFixed20_44DeclaredMultiplyOps() {
  return BuildDeclaredMultiplyOps<20u, 44u, i64>();
}

[[nodiscard]] inline auto BuildFixedLane32ArithmeticOps() {
  i32 mode[4]{};
  i32 lhs[4]{};
  i32 rhs[4]{};
  i32 addend[4]{};
  i32 out[4]{};
  const auto body = rund::compute_dsl::bind(4u)
                        .fixed<1, 31>()
                        .read<"mode">(mode)
                        .read<"lhs">(lhs)
                        .read<"rhs">(rhs)
                        .read<"addend">(addend)
                        .write<"out">(out);

  return rund::compute_dsl::def("lower-fixed-arithmetic-lane32")
      .on(body)
      .map([](auto i, auto b) {
        auto mode = b.template read<"mode">();
        auto lhs = b.template read<"lhs">();
        auto rhs = b.template read<"rhs">();
        auto addend = b.template read<"addend">();
        auto out = b.template write<"out">();

        auto result =
            rund::compute_dsl::mul_add_fixed(lhs[i], rhs[i], addend[i]);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], 10),
            rund::compute_dsl::lerp(lhs[i], rhs[i], addend[i], lhs[i],
                                    addend[i], rhs[i]),
            result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], 9),
            rund::compute_dsl::step(lhs[i], rhs[i]), result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], 8),
            rund::compute_dsl::saturate(addend[i]), result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], 7),
            rund::compute_dsl::lerp(lhs[i], rhs[i], addend[i]), result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], 6),
            rund::compute_dsl::mul_unsigned_fixed(lhs[i], rhs[i]), result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], 5),
            rund::compute_dsl::mul_fixed_scaled(lhs[i], rhs[i]), result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], 4),
            rund::compute_dsl::mul_fixed(lhs[i], rhs[i]), result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], 3),
            rund::compute_dsl::neg_positive_fixed(lhs[i]), result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], 2),
            rund::compute_dsl::sub_sat(lhs[i], rhs[i]), result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], 1),
            rund::compute_dsl::add_sat_unsigned(lhs[i], rhs[i]), result);
        out[i] = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], 0),
            rund::compute_dsl::add_sat(lhs[i], rhs[i]), result);
      });
}

[[nodiscard]] inline auto BuildFixedLane64ArithmeticOps() {
  i64 mode[4]{};
  i64 lhs[4]{};
  i64 rhs[4]{};
  i64 addend[4]{};
  i64 out[4]{};
  const auto body = rund::compute_dsl::bind(4u)
                        .fixed<1, 63>()
                        .read<"mode">(mode)
                        .read<"lhs">(lhs)
                        .read<"rhs">(rhs)
                        .read<"addend">(addend)
                        .write<"out">(out);

  return rund::compute_dsl::def("lower-fixed-arithmetic-lane64")
      .on(body)
      .map([](auto i, auto b) {
        auto mode = b.template read<"mode">();
        auto lhs = b.template read<"lhs">();
        auto rhs = b.template read<"rhs">();
        auto addend = b.template read<"addend">();
        auto out = b.template write<"out">();

        auto result =
            rund::compute_dsl::mul_add_fixed(lhs[i], rhs[i], addend[i]);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], i64{10}),
            rund::compute_dsl::lerp(lhs[i], rhs[i], addend[i], lhs[i],
                                    addend[i], rhs[i]),
            result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], i64{9}),
            rund::compute_dsl::step(lhs[i], rhs[i]), result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], i64{8}),
            rund::compute_dsl::saturate(addend[i]), result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], i64{7}),
            rund::compute_dsl::lerp(lhs[i], rhs[i], addend[i]), result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], i64{6}),
            rund::compute_dsl::mul_unsigned_fixed(lhs[i], rhs[i]), result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], i64{5}),
            rund::compute_dsl::mul_fixed_scaled(lhs[i], rhs[i]), result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], i64{4}),
            rund::compute_dsl::mul_fixed(lhs[i], rhs[i]), result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], i64{3}),
            rund::compute_dsl::neg_positive_fixed(lhs[i]), result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], i64{2}),
            rund::compute_dsl::sub_sat(lhs[i], rhs[i]), result);
        result = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], i64{1}),
            rund::compute_dsl::add_sat_unsigned(lhs[i], rhs[i]), result);
        out[i] = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], i64{0}),
            rund::compute_dsl::add_sat(lhs[i], rhs[i]), result);
      });
}

} // namespace program_compute_contract::lowering_support
