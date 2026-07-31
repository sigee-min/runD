#include "../../local.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <bit>
#include <cstdio>

namespace node_accel_contract {

[[nodiscard]] bool RunsFixedLane32ScalarOps() {
  constexpr std::size_t kTileCount = 8u;
  constexpr rund::kernel::i32 marker = 7;
  std::array<rund::kernel::i32, kTileCount> lhs{-32, -7, 0,    7,
                                                11,  64, -128, 255};
  std::array<rund::kernel::i32, kTileCount> rhs{5, -9, 0, 13, -21, 34, -55, 89};
  std::array<rund::kernel::i32, kTileCount> out{};

  const auto body = rund::compute_dsl::bind(kTileCount)
                        .fixed<1, 31>()
                        .param<"marker">(marker)
                        .read<"lhs">(lhs.data())
                        .read<"rhs">(rhs.data())
                        .write<"out">(out.data());
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-fixed-scalar-32")
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

  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const rund::kernel::BindingSet bindings = op.bindings<rund::kernel::i32>(
      21u, caps.fixed_lane32_lanes, rund::kernel::ComputeApi::Metal);
  const rund::node::accel::CpuSimdRunResult run =
      rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings);

  TEST_ASSERT(run.ok);
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    const rund::kernel::i32 sum =
        rund::math32::detail::ScalarAddWrap(lhs[index], 3);
    const rund::kernel::i32 diff =
        rund::math32::detail::ScalarSubWrap(5, rhs[index]);
    const rund::kernel::i64 product = static_cast<rund::kernel::i64>(sum) * 2;
    const rund::kernel::i64 aligned_diff =
        static_cast<rund::kernel::i64>(diff) * (rund::kernel::i64{1} << 31u);
    const bool comparison = (lhs[index] != marker) &&
                            ((product > aligned_diff) || (rhs[index] >= 0));
    const rund::kernel::i32 unary = rund::math32::detail::ScalarAddWrap(
        rund::math32::detail::ScalarAddWrap(
            rund::math32::detail::ScalarAddWrap(
                rund::math32::detail::ScalarAddWrap(
                    rund::math32::detail::ScalarSubWrap(0, lhs[index]),
                    rund::math32::detail::ScalarSubWrap(0, rhs[index])),
                rund::math32::detail::ScalarAbs(lhs[index])),
            std::bit_cast<rund::kernel::i32>(
                rund::math32::detail::ScalarAbsMagnitude(rhs[index]))),
        rund::math32::detail::ScalarSign(rhs[index]));
    const rund::kernel::i32 expected = !comparison ? unary : 0;
    if (out[index] != expected) {
      std::fprintf(stderr,
                   "ops32 scalar mismatch index=%zu lhs=%d rhs=%d "
                   "comparison=%d actual=%d expected=%d\n",
                   index, lhs[index], rhs[index], comparison ? 1 : 0,
                   out[index], expected);
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool CpuSimdUsesFullWideFixedPredicateTruthiness() {
  constexpr std::size_t kTileCount = 4u;
  std::array<rund::kernel::i32, kTileCount> input{
      std::bit_cast<rund::kernel::i32>(rund::kernel::u32{0x80000000u}), 0, 1,
      std::bit_cast<rund::kernel::i32>(rund::kernel::u32{0x7fffffffu})};
  std::array<rund::kernel::i32, kTileCount> direct{};
  std::array<rund::kernel::i32, kTileCount> connected{};

  const auto body = rund::compute_dsl::bind(kTileCount)
                        .fixed<16, 16>()
                        .read<"input">(input.data())
                        .write<"direct">(direct.data())
                        .write<"connected">(connected.data());
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-wide-predicate-truthiness")
          .on(body)
          .map([](auto i, auto b) {
            auto input = b.template read<"input">();
            auto direct = b.template write<"direct">();
            auto connected = b.template write<"connected">();
            const auto wide = input[i] + input[i];
            const auto zero = input[i] - input[i];
            direct[i] = rund::compute_dsl::select(wide, 1, 0);
            connected[i] = rund::compute_dsl::select(
                rund::compute_dsl::predicate_and(
                    rund::compute_dsl::predicate_or(wide, zero),
                    rund::compute_dsl::predicate_not(zero)),
                1, 0);
          });

  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const rund::kernel::BindingSet bindings = op.bindings<rund::kernel::i32>(
      22u, caps.fixed_lane32_lanes, rund::kernel::ComputeApi::Metal);
  const rund::node::accel::CpuSimdRunResult run =
      rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings);

  TEST_ASSERT(run.ok);
  constexpr std::array<rund::kernel::i32, kTileCount> expected{1, 0, 1, 1};
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    if (direct[index] != expected[index] ||
        connected[index] != expected[index]) {
      std::fprintf(stderr,
                   "wide predicate mismatch index=%zu input=%d direct=%d "
                   "connected=%d expected=%d\n",
                   index, input[index], direct[index], connected[index],
                   expected[index]);
      return false;
    }
  }
  return true;
}

} // namespace node_accel_contract
