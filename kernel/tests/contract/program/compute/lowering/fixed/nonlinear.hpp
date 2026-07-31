#pragma once

#include "contract/program/compute/lowering/bytes.hpp"
#include "contract/program/compute/lowering/ops.hpp"

namespace program_compute_contract::nonlinear_support {

using namespace lowering_support;

[[nodiscard]] inline auto BuildFixedLane32NonlinearOps() {
  i32 mode[4]{};
  i32 lhs[4]{};
  i32 rhs[4]{};
  i32 out[4]{};
  const auto body =
      rund::compute_dsl::bind(4u)
          .fixed<1, 31, rund::kernel::ComputeRounding::NearestEven,
                 rund::kernel::ComputeOverflow::Saturate,
                 rund::kernel::ComputeApproximation::Deterministic>()
          .read<"mode">(mode)
          .read<"lhs">(lhs)
          .read<"rhs">(rhs)
          .write<"out">(out);

  return rund::compute_dsl::def("lower-fixed-nonlinear-lane32")
      .on(body)
      .map([](auto i, auto b) {
        auto mode = b.template read<"mode">();
        auto lhs = b.template read<"lhs">();
        auto rhs = b.template read<"rhs">();
        auto out = b.template write<"out">();

        auto result = rund::compute_dsl::rsqrt(lhs[i]);
        result =
            rund::compute_dsl::select(rund::compute_dsl::eq(mode[i], 2),
                                      rund::compute_dsl::sqrt(lhs[i]), result);
        result =
            rund::compute_dsl::select(rund::compute_dsl::eq(mode[i], 1),
                                      rund::compute_dsl::recip(lhs[i]), result);
        out[i] = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], 0),
            rund::compute_dsl::div_fixed(lhs[i], rhs[i]), result);
      });
}

[[nodiscard]] inline auto BuildFixedLane64NonlinearOps() {
  i64 mode[4]{};
  i64 lhs[4]{};
  i64 rhs[4]{};
  i64 out[4]{};
  const auto body =
      rund::compute_dsl::bind(4u)
          .fixed<1, 63, rund::kernel::ComputeRounding::NearestEven,
                 rund::kernel::ComputeOverflow::Saturate,
                 rund::kernel::ComputeApproximation::Deterministic>()
          .read<"mode">(mode)
          .read<"lhs">(lhs)
          .read<"rhs">(rhs)
          .write<"out">(out);

  return rund::compute_dsl::def("lower-fixed-nonlinear-lane64")
      .on(body)
      .map([](auto i, auto b) {
        auto mode = b.template read<"mode">();
        auto lhs = b.template read<"lhs">();
        auto rhs = b.template read<"rhs">();
        auto out = b.template write<"out">();

        auto result = rund::compute_dsl::rsqrt(lhs[i]);
        result =
            rund::compute_dsl::select(rund::compute_dsl::eq(mode[i], i64{2}),
                                      rund::compute_dsl::sqrt(lhs[i]), result);
        result =
            rund::compute_dsl::select(rund::compute_dsl::eq(mode[i], i64{1}),
                                      rund::compute_dsl::recip(lhs[i]), result);
        out[i] = rund::compute_dsl::select(
            rund::compute_dsl::eq(mode[i], i64{0}),
            rund::compute_dsl::div_fixed(lhs[i], rhs[i]), result);
      });
}

[[nodiscard]] inline std::vector<rund::kernel::u8> FixedNonlinearUnaryIrBytes(
    const rund::kernel::IrOp op, const rund::kernel::u32 lhs,
    const rund::kernel::u32 rhs, const rund::kernel::u32 aux) {
  std::vector<rund::kernel::u8> bytes;
  AppendBytes(bytes, "rund.compute.ir");
  AppendBytes(bytes, "forged-fixed-nonlinear-unary");
  AppendU8(bytes, lowering_support::kI32NumericMode);
  AppendIntegerNumericPolicy(bytes);
  AppendU32(bytes, 3u);

  AppendBinding(bytes, 2u, "lhs");
  AppendBinding(bytes, 2u, "rhs");
  AppendBinding(bytes, 3u, "out");

  AppendU32(bytes, 4u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 0u);
  AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, 1u);
  AppendNode(bytes, op, lhs, rhs, aux);
  AppendNode(bytes, rund::kernel::IrOp::Write, 3u, 0u, 2u);
  return bytes;
}

} // namespace program_compute_contract::nonlinear_support
