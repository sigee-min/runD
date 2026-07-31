#pragma once

#include "base.hpp"

namespace program_compute_contract::lowering_support {

[[nodiscard]] inline auto BuildFixedLane32ExpandedOps() {
  i32 lhs[4]{};
  i32 rhs[4]{};
  i32 out[4]{};
  const auto body = rund::compute_dsl::bind(4u)
                        .fixed<1, 31>()
                        .param<"scale">(3)
                        .param<"bias">(-17)
                        .param<"lo">(-120)
                        .param<"hi">(120)
                        .param<"marker">(7)
                        .read<"lhs">(lhs)
                        .read<"rhs">(rhs)
                        .write<"out">(out);

  return rund::compute_dsl::def("lower-fixed_lane32-expanded")
      .on(body)
      .map([](auto i, auto b) {
        auto scale = b.template param<"scale">();
        auto bias = b.template param<"bias">();
        auto lo = b.template param<"lo">();
        auto hi = b.template param<"hi">();
        auto marker = b.template param<"marker">();
        auto lhs = b.template read<"lhs">();
        auto rhs = b.template read<"rhs">();
        auto out = b.template write<"out">();

        auto mixed = (lhs[i] * scale) + bias - rhs[i];
        auto bounded = rund::compute_dsl::clamp(mixed, lo, hi);
        auto signed_extreme =
            rund::compute_dsl::min(rund::compute_dsl::max(lhs[i], rhs[i]), hi);
        out[i] = rund::compute_dsl::select(
            rund::compute_dsl::eq(lhs[i], marker), rund::compute_dsl::le(rhs[i], lhs[i]),
            rund::compute_dsl::select(rund::compute_dsl::lt(lhs[i], rhs[i]), bounded,
                                signed_extreme));
      });
}

[[nodiscard]] inline auto BuildFixedLane64ExpandedOps() {
  i64 lhs[4]{};
  i64 rhs[4]{};
  i64 out[4]{};
  const auto body = rund::compute_dsl::bind(4u)
                        .fixed<1, 63>()
                        .param<"scale">(i64{0x100000003ll})
                        .param<"bias">(i64{-0x123456789ll})
                        .param<"lo">(i64{-0x4000000000ll})
                        .param<"hi">(i64{0x4000000000ll})
                        .param<"marker">(i64{7})
                        .read<"lhs">(lhs)
                        .read<"rhs">(rhs)
                        .write<"out">(out);

  return rund::compute_dsl::def("lower-fixed_lane64-expanded")
      .on(body)
      .map([](auto i, auto b) {
        auto scale = b.template param<"scale">();
        auto bias = b.template param<"bias">();
        auto lo = b.template param<"lo">();
        auto hi = b.template param<"hi">();
        auto marker = b.template param<"marker">();
        auto lhs = b.template read<"lhs">();
        auto rhs = b.template read<"rhs">();
        auto out = b.template write<"out">();

        auto mixed = rund::compute_dsl::quantize<1, 63>(lhs[i] * scale) +
                     bias - rhs[i];
        auto bounded = rund::compute_dsl::clamp(mixed, lo, hi);
        auto signed_extreme =
            rund::compute_dsl::min(rund::compute_dsl::max(lhs[i], rhs[i]), hi);
        out[i] = rund::compute_dsl::select(
            rund::compute_dsl::eq(lhs[i], marker), rund::compute_dsl::le(rhs[i], lhs[i]),
            rund::compute_dsl::select(rund::compute_dsl::lt(lhs[i], rhs[i]), bounded,
                                signed_extreme));
      });
}

} // namespace program_compute_contract::lowering_support
