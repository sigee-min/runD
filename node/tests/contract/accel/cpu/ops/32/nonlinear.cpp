#include "../../fixed.hpp"
#include "../../local.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>

namespace node_accel_contract {

[[nodiscard]] bool RunsFixedLane32NonlinearOps() {
  constexpr std::size_t kTileCount = 8u;
  std::array<rund::kernel::i32, kTileCount> mode{0, 1, 2, 3, 0, 1, 2, 3};
  std::array<rund::kernel::i32, kTileCount> lhs{
      0x40000000,  0x20000000, 0x10000000, 0x08000000,
      -0x30000000, 0,          0x01000000, 0x7fffffff};
  std::array<rund::kernel::i32, kTileCount> rhs{
      0x20000000, 0x10000000, 0x40000000, 0x08000000,
      0,          0x10000000, 0x01000000, -0x20000000};
  std::array<rund::kernel::i32, kTileCount> out{};

  const auto body = rund::compute_dsl::bind(kTileCount)
                        .fixed<1, 31>()
                        .read<"mode">(mode.data())
                        .read<"lhs">(lhs.data())
                        .read<"rhs">(rhs.data())
                        .write<"out">(out.data());
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-fixed-nonlinear-32")
          .on(body)
          .map([](auto i, auto b) {
            auto mode = b.template read<"mode">();
            auto lhs = b.template read<"lhs">();
            auto rhs = b.template read<"rhs">();
            auto out = b.template write<"out">();

            auto result = rund::compute_dsl::rsqrt(lhs[i]);
            result = rund::compute_dsl::select(
                rund::compute_dsl::eq(mode[i], 2),
                rund::compute_dsl::sqrt(lhs[i]), result);
            result = rund::compute_dsl::select(
                rund::compute_dsl::eq(mode[i], 1),
                rund::compute_dsl::recip(lhs[i]), result);
            out[i] = rund::compute_dsl::select(
                rund::compute_dsl::eq(mode[i], 0),
                rund::compute_dsl::div_fixed(lhs[i], rhs[i]), result);
          });

  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const rund::kernel::BindingSet bindings = op.bindings<rund::kernel::i32>(
      24u, caps.fixed_lane32_lanes, rund::kernel::ComputeApi::Metal);
  const rund::node::accel::CpuSimdRunResult run =
      rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings);

  TEST_ASSERT(run.ok);
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    rund::kernel::i32 expected = cpu::fixed::RsqrtNearestEven(lhs[index]);
    if (mode[index] == 2) {
      expected = cpu::fixed::SqrtFloor(lhs[index]);
    } else if (mode[index] == 1) {
      expected = cpu::fixed::RecipNearestEven(lhs[index]);
    } else if (mode[index] == 0) {
      expected = cpu::fixed::DivNearestEven(lhs[index], rhs[index]);
    }
    TEST_ASSERT(out[index] == expected);
  }
  return true;
}

} // namespace node_accel_contract
