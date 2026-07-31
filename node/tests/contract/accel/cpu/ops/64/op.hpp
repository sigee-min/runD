#pragma once

#include "work.hpp"

namespace node_accel_contract::cpu::ops64 {

[[nodiscard]] rund::compute_dsl::ComputeOp BuildOp(Work &work) {
  const auto body = rund::compute_dsl::bind(kTileCount)
                        .fixed<1, 63>()
                        .read<"lhs">(work.lhs.data())
                        .read<"rhs">(work.rhs.data())
                        .read<"lo">(work.lo.data())
                        .read<"hi">(work.hi.data())
                        .read<"addend">(work.addend.data())
                        .read<"positive">(work.positive.data())
                        .write<"out">(work.out.data());
  return rund::compute_dsl::def("node-cpu-simd-i1-f63")
      .on(body)
      .map([](auto i, auto b) {
        auto lhs = b.template read<"lhs">();
        auto rhs = b.template read<"rhs">();
        auto lo = b.template read<"lo">();
        auto hi = b.template read<"hi">();
        auto addend = b.template read<"addend">();
        auto positive = b.template read<"positive">();
        auto out = b.template write<"out">();

        auto ordered = rund::compute_dsl::predicate_and(
            rund::compute_dsl::predicate_or(
                rund::compute_dsl::lt(lhs[i], rhs[i]),
                rund::compute_dsl::le(lhs[i], rhs[i])),
            rund::compute_dsl::predicate_or(
                rund::compute_dsl::predicate_or(
                    rund::compute_dsl::eq(lhs[i], rhs[i]),
                    rund::compute_dsl::ne(lhs[i], rhs[i])),
                rund::compute_dsl::predicate_and(
                    rund::compute_dsl::gt(hi[i], lo[i]),
                    rund::compute_dsl::ge(hi[i], lhs[i]))));
        auto scalar = rund::compute_dsl::select(
            rund::compute_dsl::predicate_not(ordered),
            rund::compute_dsl::clamp(rund::compute_dsl::min(lhs[i], rhs[i]) +
                                         rund::compute_dsl::max(lhs[i], rhs[i]),
                                     lo[i], hi[i]) +
                rund::compute_dsl::neg(lhs[i]) + (-rhs[i]) +
                rund::compute_dsl::abs(lhs[i]) +
                rund::compute_dsl::abs_magnitude(rhs[i]) +
                rund::compute_dsl::sign(rhs[i]),
            0);
        auto bits = rund::compute_dsl::bit_or(
            rund::compute_dsl::bit_and(lhs[i], rhs[i]),
            rund::compute_dsl::bit_xor(
                rund::compute_dsl::bit_not(lhs[i]),
                rund::compute_dsl::shl_const<5>(rhs[i])));
        bits = bits + rund::compute_dsl::shr_logical_const<9>(lhs[i]) +
               rund::compute_dsl::shr_arithmetic_const<11>(rhs[i]);
        auto arithmetic =
            rund::compute_dsl::add_sat(lhs[i], rhs[i]) +
            rund::compute_dsl::add_sat_unsigned(lhs[i], rhs[i]) +
            rund::compute_dsl::sub_sat(lhs[i], rhs[i]) +
            rund::compute_dsl::neg_positive_fixed(positive[i]) +
            rund::compute_dsl::mul_fixed(lhs[i], rhs[i]) +
            rund::compute_dsl::mul_fixed_scaled(lhs[i], positive[i]) +
            rund::compute_dsl::mul_unsigned_fixed(positive[i], rhs[i]) +
            rund::compute_dsl::detail::StorageQuantize(
                rund::compute_dsl::mul_add_fixed(lhs[i], rhs[i], addend[i])) +
            rund::compute_dsl::saturate(positive[i]) +
            rund::compute_dsl::step(lhs[i], rhs[i]) +
            rund::compute_dsl::lerp(lhs[i], rhs[i], positive[i]) +
            rund::compute_dsl::lerp(lhs[i], rhs[i], lo[i], hi[i], positive[i],
                                    positive[i]);
        auto nonlinear = rund::compute_dsl::div_fixed(lhs[i], rhs[i]) +
                         rund::compute_dsl::recip(positive[i]) +
                         rund::compute_dsl::sqrt(positive[i]) +
                         rund::compute_dsl::rsqrt(positive[i]);
        scalar = rund::compute_dsl::detail::StorageQuantize(scalar);
        bits = rund::compute_dsl::detail::StorageQuantize(bits);
        arithmetic = rund::compute_dsl::detail::StorageQuantize(arithmetic);
        nonlinear = rund::compute_dsl::detail::StorageQuantize(nonlinear);
        out[i] = rund::compute_dsl::detail::StorageQuantize(
            scalar + bits + arithmetic + nonlinear);
      });
}

} // namespace node_accel_contract::cpu::ops64
