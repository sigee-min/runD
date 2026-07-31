#include "../../fixed.hpp"
#include "../../local.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <bit>

namespace node_accel_contract {

[[nodiscard]] rund::kernel::i32
ScalarLerp32(const rund::kernel::i32 lhs, const rund::kernel::i32 rhs,
             const rund::kernel::i32 amount) noexcept {
  constexpr rund::kernel::i32 fixed_max = 0x7fffffff;
  const rund::kernel::i32 t =
      rund::math32::detail::ScalarClamp(amount, 0, fixed_max);
  return rund::math32::detail::ScalarAddSat(
      cpu::fixed::QuantizeProduct(
          lhs, rund::math32::detail::ScalarSubSat(fixed_max, t)),
      cpu::fixed::QuantizeProduct(rhs, t));
}

[[nodiscard]] bool RunsFixedLane32ArithmeticOps() {
  constexpr std::size_t kTileCount = 8u;
  std::array<rund::kernel::i32, kTileCount> mode{0, 1, 2, 3, 4, 5, 6, 7};
  std::array<rund::kernel::i32, kTileCount> lhs{
      0x10000000,  -0x20000000, 0x30000000, 0x7fffffff,
      -0x40000000, 123456789,   -7654321,   3333333};
  std::array<rund::kernel::i32, kTileCount> rhs{
      0x08000000,  0x70000000, -0x10000000, 0x01000000,
      -0x20000000, 0x12345678, 0x0fffffff,  -333333};
  std::array<rund::kernel::i32, kTileCount> addend{1, -2, 3, -4,
                                                   5, -6, 7, 0x40000000};
  std::array<rund::kernel::i32, kTileCount> out{};

  const auto body = rund::compute_dsl::bind(kTileCount)
                        .fixed<1, 31>()
                        .read<"mode">(mode.data())
                        .read<"lhs">(lhs.data())
                        .read<"rhs">(rhs.data())
                        .read<"addend">(addend.data())
                        .write<"out">(out.data());
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-fixed-arithmetic-32")
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

  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const rund::kernel::BindingSet bindings = op.bindings<rund::kernel::i32>(
      23u, caps.fixed_lane32_lanes, rund::kernel::ComputeApi::Metal);
  const rund::node::accel::CpuSimdRunResult run =
      rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings);

  TEST_ASSERT(run.ok);
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    rund::kernel::i32 expected =
        cpu::fixed::QuantizeMulAdd(lhs[index], rhs[index], addend[index]);
    if (mode[index] == 6) {
      expected =
          std::bit_cast<rund::kernel::i32>(cpu::fixed::QuantizeUnsignedProduct(
              std::bit_cast<rund::kernel::u32>(lhs[index]),
              std::bit_cast<rund::kernel::u32>(rhs[index])));
    } else if (mode[index] == 7) {
      expected = ScalarLerp32(lhs[index], rhs[index], addend[index]);
    } else if (mode[index] == 5) {
      expected = cpu::fixed::QuantizeScaledProduct(
          lhs[index], std::bit_cast<rund::kernel::u32>(rhs[index]));
    } else if (mode[index] == 4) {
      expected = cpu::fixed::QuantizeProduct(lhs[index], rhs[index]);
    } else if (mode[index] == 3) {
      expected = rund::math32::detail::ScalarNegPositiveFixed(lhs[index]);
    } else if (mode[index] == 2) {
      expected = rund::math32::detail::ScalarSubSat(lhs[index], rhs[index]);
    } else if (mode[index] == 1) {
      expected = std::bit_cast<rund::kernel::i32>(
          rund::math32::detail::ScalarAddSatUnsigned(
              std::bit_cast<rund::kernel::u32>(lhs[index]),
              std::bit_cast<rund::kernel::u32>(rhs[index])));
    } else if (mode[index] == 0) {
      expected = rund::math32::detail::ScalarAddSat(lhs[index], rhs[index]);
    }
    TEST_ASSERT(out[index] == expected);
  }

  return true;
}

} // namespace node_accel_contract
