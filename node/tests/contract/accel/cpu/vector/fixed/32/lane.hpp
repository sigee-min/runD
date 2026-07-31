#pragma once

#include <node/accel/cpu/simd.hpp>

#include "../../../fixed.hpp"
#include "../../../local.hpp"

#include <array>
#include <string_view>

namespace node_accel_contract {

[[nodiscard]] bool RunsFixedLane32MapWithVectorLaw() {
  constexpr std::size_t kTileCount = 8u;
  constexpr rund::kernel::i32 dt = 7;
  std::array<rund::kernel::i32, kTileCount> pos{1, 2, -3, 4, -5, 6, 7, -8};
  std::array<rund::kernel::i32, kTileCount> vel{3, -4, 5, -6, 7, -8, 9, 10};
  std::array<rund::kernel::i32, kTileCount> out{};
  const auto body = rund::compute_dsl::bind(kTileCount)
                        .fixed<1, 31>()
                        .param<"dt">(dt)
                        .read<"pos">(pos.data())
                        .read<"vel">(vel.data())
                        .write<"out">(out.data());
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-fixed_lane32")
          .on(body)
          .map([](auto i, auto b) {
            auto dt_value = b.template param<"dt">();
            auto pos = b.template read<"pos">();
            auto vel = b.template read<"vel">();
            auto out = b.template write<"out">();
            out[i] = pos[i] + (vel[i] * dt_value);
          });
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const rund::kernel::BindingSet bindings = op.bindings<rund::kernel::i32>(
      17u, caps.fixed_lane32_lanes, rund::kernel::ComputeApi::Metal);
  const rund::node::accel::CpuSimdRunResult run =
      rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings);
  TEST_ASSERT(run.ok);
  TEST_ASSERT(std::string_view{run.reason} == "ok");
  TEST_ASSERT(run.strategy == rund::kernel::CpuSimdStrategy::Neon);
  TEST_ASSERT(run.processed_tiles == kTileCount);
  TEST_ASSERT(run.vector_chunk_count == 2u);
  TEST_ASSERT(run.tail_chunk_count == 0u);
  TEST_ASSERT(run.rejected_count == 0u);
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    TEST_ASSERT(out[index] ==
                cpu::fixed::QuantizeMulAdd(vel[index], dt, pos[index]));
  }
  return true;
}

} // namespace node_accel_contract
