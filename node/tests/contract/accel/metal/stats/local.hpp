#pragma once

#include <kernel/program/compute/dsl.hpp>
#include <kernel/program/phase.hpp>
namespace node_accel_contract::metal_stats {

struct NodeComputeOp {
  rund::kernel::ComputeIR ir{};
  rund::kernel::ComputeMap map{};
};

[[nodiscard]] inline rund::kernel::TilePhaseDescription Phase() {
  return rund::kernel::TilePhaseDescription{
      .phase_id = 44u,
      .tile_count = 4u,
      .capacity =
          rund::kernel::TilePhaseCapacityRequirement{
              .scratch_bytes_per_tile = 0u,
              .scratch_alignment = 1u,
              .output_shards = 4u,
              .queue_slots = 4u,
              .task_slots = 4u,
          },
  };
}

[[nodiscard]] inline NodeComputeOp BuildOp(
    const rund::kernel::ComputeApi api) {
  rund::kernel::i32 lhs[4]{};
  rund::kernel::i32 rhs[4]{};
  rund::kernel::i32 output[4]{};
  const auto body = rund::compute_dsl::bind(4u)
                        .fixed<1, 31>()
                        .param<"dt">(7)
                        .read<"lhs">(lhs)
                        .read<"rhs">(rhs)
                        .write<"output">(output);
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-metal-stats-accel").on(body).map(
          [](auto i, auto b) {
            auto dt = b.template param<"dt">();
            auto lhs = b.template read<"lhs">();
            auto rhs = b.template read<"rhs">();
            auto output = b.template write<"output">();
            output[i] = lhs[i] + rhs[i] + dt;
          });
  rund::kernel::ComputeMap map = op.map();
  map.api = api;
  return NodeComputeOp{.ir = op.ir(), .map = map};
}

}  // namespace node_accel_contract::metal_stats
