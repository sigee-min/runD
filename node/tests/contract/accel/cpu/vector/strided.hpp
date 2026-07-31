#pragma once

#include <node/accel/cpu/simd.hpp>

#include "../local.hpp"

#include <array>

namespace node_accel_contract {

[[nodiscard]] bool RunsFixedLane32StridedBindingsWithVectorLaw() {
  constexpr std::size_t kTileCount = 9u;
  constexpr rund::kernel::i32 bias = 5;
  constexpr rund::kernel::i32 gap = -7777;
  constexpr std::array<rund::kernel::i32, kTileCount> values{
      2, -4, 8, -16, 32, -64, 128, -256, 512};
  std::array<rund::kernel::i32, kTileCount * 2u> input{};
  std::array<rund::kernel::i32, kTileCount * 2u> out{};
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    input[index * 2u] = values[index];
    input[index * 2u + 1u] = gap;
    out[index * 2u] = gap;
    out[index * 2u + 1u] = gap;
  }
  const auto body = rund::compute_dsl::bind(kTileCount)
                        .fixed<1, 31>()
                        .param<"bias">(bias)
                        .read<"input">(input.data())
                        .write<"out">(out.data());
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-strided")
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
  rund::kernel::BindingSet bindings = op.bindings<rund::kernel::i32>(
      28u, caps.fixed_lane32_lanes, rund::kernel::ComputeApi::Metal);
  std::array<rund::kernel::BufferSpan, 1u> inputs{bindings.input_buffers[0]};
  inputs[0].stride_bytes = sizeof(rund::kernel::i32) * 2u;
  bindings.input_buffers = inputs.data();
  std::array<rund::kernel::OutputSpan, 1u> outputs{bindings.output_buffers[0]};
  outputs[0].stride_bytes = sizeof(rund::kernel::i32) * 2u;
  bindings.output_buffers = outputs.data();
  const rund::node::accel::CpuSimdRunResult run =
      rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings);
  TEST_ASSERT(run.ok);
  TEST_ASSERT(run.processed_tiles == kTileCount);
  TEST_ASSERT(run.vector_chunk_count == 2u);
  TEST_ASSERT(run.tail_chunk_count == 1u);
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    TEST_ASSERT(out[index * 2u] == values[index] + bias);
    TEST_ASSERT(input[index * 2u + 1u] == gap);
    TEST_ASSERT(out[index * 2u + 1u] == gap);
  }
  return true;
}

} // namespace node_accel_contract
