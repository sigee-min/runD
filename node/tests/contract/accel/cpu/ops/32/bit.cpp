#include "../../local.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <bit>

namespace node_accel_contract {

[[nodiscard]] bool RunsFixedLane32BitOps() {
  constexpr std::size_t kTileCount = 8u;
  std::array<rund::kernel::i32, kTileCount> lhs{
      0x01020304, -1, 0x40000000, -0x40000000, 12345, -54321, 7, -8};
  std::array<rund::kernel::i32, kTileCount> rhs{
      0x11111111, 3, -9, 0x22222222, -333, 444, -555, 666};
  std::array<rund::kernel::i32, kTileCount> out{};

  const auto body = rund::compute_dsl::bind(kTileCount)
                        .fixed<1, 31>()
                        .read<"lhs">(lhs.data())
                        .read<"rhs">(rhs.data())
                        .write<"out">(out.data());
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-fixed-bit-32")
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

  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const rund::kernel::BindingSet bindings = op.bindings<rund::kernel::i32>(
      22u, caps.fixed_lane32_lanes, rund::kernel::ComputeApi::Metal);
  const rund::node::accel::CpuSimdRunResult run =
      rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings);

  TEST_ASSERT(run.ok);
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    const auto lhs_bits = std::bit_cast<rund::kernel::u32>(lhs[index]);
    const auto rhs_bits = std::bit_cast<rund::kernel::u32>(rhs[index]);
    const rund::kernel::i32 mixed = std::bit_cast<rund::kernel::i32>(
        (lhs_bits & rhs_bits) | ((~lhs_bits) ^ (rhs_bits << 3u)));
    const rund::kernel::i32 expected = rund::math32::detail::ScalarAddWrap(
        rund::math32::detail::ScalarAddWrap(
            mixed, cpu::LogicalShiftRight32(lhs[index], 5u)),
        cpu::ArithmeticShiftRight32(rhs[index], 7u));
    TEST_ASSERT(out[index] == expected);
  }

  return true;
}

} // namespace node_accel_contract
