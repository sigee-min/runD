#pragma once

#include "work.hpp"

namespace node_accel_contract::cpu::pick {

[[nodiscard]] inline rund::compute_dsl::ComputeOp BuildOp(Work &work) {
  const auto body = rund::compute_dsl::bind(kTileCount)
                        .fixed<1, 31>()
                        .read<"lhs">(work.lhs.data())
                        .read<"rhs">(work.rhs.data())
                        .write<"out">(work.out.data());
  return rund::compute_dsl::def("node-cpu-generic-backend")
      .on(body)
      .map([](auto i, auto b) {
        auto lhs = b.template read<"lhs">();
        auto rhs = b.template read<"rhs">();
        auto out = b.template write<"out">();

        out[i] = rund::compute_dsl::quantize<
            1u, 31u, rund::kernel::ComputeRounding::NearestEven,
            rund::kernel::ComputeOverflow::Saturate,
            rund::kernel::ComputeApproximation::Deterministic>(
            rund::compute_dsl::div_fixed(lhs[i], rhs[i]) +
            rund::compute_dsl::sqrt(rund::compute_dsl::quantize<1u, 31u>(
                rund::compute_dsl::abs(lhs[i]))) +
            rund::compute_dsl::select(
                rund::compute_dsl::gt(lhs[i], rhs[i]),
                rund::compute_dsl::add_sat(lhs[i], rhs[i]),
                rund::compute_dsl::sub_sat(lhs[i], rhs[i])));
      });
}

} // namespace node_accel_contract::cpu::pick
