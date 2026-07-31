#pragma once

#include <node/accel/cpu/simd.hpp>

#include "../local.hpp"

#include <array>

namespace node_accel_contract {

[[nodiscard]] bool RunsFixedLane32TailChunkWithVectorLaw() {
  constexpr std::size_t kTileCount = 6u;
  constexpr rund::kernel::i32 bias = -3;
  std::array<rund::kernel::i32, kTileCount> input{5, -7, 11, -13, 17, -19};
  std::array<rund::kernel::i32, kTileCount> out{};
  const auto body = rund::compute_dsl::bind(kTileCount)
                        .fixed<1, 31>()
                        .param<"bias">(bias)
                        .read<"input">(input.data())
                        .write<"out">(out.data());
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-tail")
          .on(body)
          .map([](auto i, auto b) {
            auto bias = b.template param<"bias">();
            auto input = b.template read<"input">();
            auto out = b.template write<"out">();
            out[i] = input[i] + bias;
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const rund::kernel::BindingSet bindings = op.bindings<rund::kernel::i32>(
      20u, caps.fixed_lane32_lanes, rund::kernel::ComputeApi::Metal);
  const rund::node::accel::CpuSimdRunResult run =
      rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings);
  TEST_ASSERT(run.ok);
  TEST_ASSERT(run.processed_tiles == kTileCount);
  TEST_ASSERT(run.vector_chunk_count == 1u);
  TEST_ASSERT(run.tail_chunk_count == 1u);
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    TEST_ASSERT(out[index] == input[index] + bias);
  }
  return true;
}

} // namespace node_accel_contract
