#include "run.hpp"
#include <node/accel/cpu/simd.hpp>

#include <array>
#include <bit>
#include <cstdio>

namespace node_accel_contract {

[[nodiscard]] bool CpuSimdUsesFullWideFixedLane64PredicateTruthiness() {
  constexpr std::size_t kTileCount = 4u;
  std::array<rund::kernel::i64, kTileCount> input{
      std::bit_cast<rund::kernel::i64>(
          rund::kernel::u64{0x8000000000000000ull}),
      0, 1,
      std::bit_cast<rund::kernel::i64>(
          rund::kernel::u64{0x7fffffffffffffffull})};
  std::array<rund::kernel::i64, kTileCount> direct{};
  std::array<rund::kernel::i64, kTileCount> connected{};

  const auto body = rund::compute_dsl::bind(kTileCount)
                        .fixed<20, 44>()
                        .read<"input">(input.data())
                        .write<"direct">(direct.data())
                        .write<"connected">(connected.data());
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-wide-lane64-predicate-truthiness")
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
  const rund::kernel::BindingSet bindings = op.bindings<rund::kernel::i64>(
      26u, caps.fixed_lane64_lanes, rund::kernel::ComputeApi::Metal);
  const rund::node::accel::CpuSimdRunResult run =
      rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings);

  TEST_ASSERT(run.ok);
  constexpr std::array<rund::kernel::i64, kTileCount> expected{1, 0, 1, 1};
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    if (direct[index] != expected[index] ||
        connected[index] != expected[index]) {
      std::fprintf(stderr,
                   "wide lane64 predicate mismatch index=%zu input=%lld "
                   "direct=%lld connected=%lld expected=%lld\n",
                   index, static_cast<long long>(input[index]),
                   static_cast<long long>(direct[index]),
                   static_cast<long long>(connected[index]),
                   static_cast<long long>(expected[index]));
      return false;
    }
  }
  return true;
}

} // namespace node_accel_contract
