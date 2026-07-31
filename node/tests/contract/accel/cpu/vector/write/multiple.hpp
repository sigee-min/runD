#pragma once

#include <node/accel/cpu/simd.hpp>

#include "../../fixed.hpp"
#include "../../local.hpp"

#include <array>

namespace node_accel_contract {

[[nodiscard]] bool RunsFixedLane32MultiWriteWithOneEvaluation() {
  constexpr std::size_t kTileCount = 9u;
  constexpr std::array<rund::kernel::i32, kTileCount> input{-7, -3, -1, 0, 2,
                                                            5,  11, 17, 29};
  std::array<rund::kernel::i32, kTileCount> plus{};
  std::array<rund::kernel::i32, kTileCount> times{};

  const auto body = rund::compute_dsl::bind(kTileCount)
                        .fixed<1, 31>()
                        .read<"input">(input.data())
                        .write<"plus">(plus.data())
                        .write<"times">(times.data());
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-write-multiple")
          .on(body)
          .map([](auto i, auto b) {
            const auto input = b.template read<"input">();
            const auto plus = b.template write<"plus">();
            const auto times = b.template write<"times">();
            plus[i] = input[i] + 1;
            times[i] = input[i] * 3;
          });

  TEST_ASSERT(op.ok());
  TEST_ASSERT(op.map().output_buffer_count == 2u);
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  TEST_ASSERT(artifact.ok);
  const rund::kernel::BindingSet bindings = op.bindings<rund::kernel::i32>(
      29u, caps.fixed_lane32_lanes, rund::kernel::ComputeApi::Cpu);
  TEST_ASSERT(bindings.output_buffer_count == 2u);
  const rund::node::accel::CpuSimdRunResult run =
      rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings);
  TEST_ASSERT(run.ok);
  TEST_ASSERT(run.processed_tiles == kTileCount);
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    TEST_ASSERT(plus[index] == input[index] + 1);
    TEST_ASSERT(times[index] == cpu::fixed::QuantizeProduct(input[index], 3));
  }
  return true;
}

} // namespace node_accel_contract
