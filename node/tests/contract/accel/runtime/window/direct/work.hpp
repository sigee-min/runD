#pragma once

#include <accel/device.hpp>

#include "../local.hpp"

#include <kernel/program/compute/dsl.hpp>
#include <kernel/program/compute/lowering/entry.hpp>
#include <kernel/program/compute/plan.hpp>

#include <array>

namespace node_accel_contract::runtime::window::direct {

inline constexpr std::size_t kTileCount = 10u;

struct Work {
  std::array<rund::kernel::u32, kTileCount> lhs{};
  std::array<rund::kernel::u32, kTileCount> rhs{};
  std::array<rund::kernel::u32, kTileCount> out{};
  std::array<rund::kernel::u32, kTileCount> expected{};
  rund::kernel::TilePhaseDescription phase{};
  rund::kernel::ComputePlan plan{};
  rund::kernel::LoweringArtifact artifact{};
};

inline void FillInputs(Work &work) {
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    work.lhs[index] = 7u + static_cast<rund::kernel::u32>(index);
    work.rhs[index] = 13u + static_cast<rund::kernel::u32>(index * 3u);
    work.expected[index] = work.lhs[index] + work.rhs[index];
  }
}

[[nodiscard]] inline rund::compute_dsl::ComputeOp BuildOp(Work &work) {
  const auto body = rund::compute_dsl::bind(kTileCount)
                        .fixed<1, 31>()
                        .read<"lhs">(work.lhs.data())
                        .read<"rhs">(work.rhs.data())
                        .write<"out">(work.out.data());
  return rund::compute_dsl::def("node-direct-resident-identity")
      .on(body)
      .map([](auto i, auto b) {
        b.template write<"out">()[i] =
            b.template read<"lhs">()[i] + b.template read<"rhs">()[i];
      });
}

[[nodiscard]] inline rund::kernel::TilePhaseDescription Phase() {
  return rund::kernel::TilePhaseDescription{
      .phase_id = 50u,
      .tile_count = kTileCount,
      .capacity =
          rund::kernel::TilePhaseCapacityRequirement{
              .scratch_bytes_per_tile = 0u,
              .scratch_alignment = 1u,
              .output_shards = kTileCount,
              .queue_slots = kTileCount,
              .task_slots = kTileCount,
          },
  };
}

[[nodiscard]] inline bool PrepareWork(const rund::AccelDevice &pick,
                                      const rund::compute_dsl::ComputeOp &op,
                                      Work &work) {
  rund::kernel::ComputeMap map = op.map();
  map.api = pick.caps.api;
  work.phase = Phase();
  work.plan =
      rund::kernel::PlanCompute(work.phase, map, pick.caps,
                                rund::kernel::ComputeLimit{
                                    .staging_bytes = pick.caps.staging_bytes,
                                    .max_window_tiles = 4u,
                                });
  work.artifact = rund::kernel::LowerComputeIR(op.ir(), work.plan.api);
  return work.plan.ok && work.plan.input_buffer_count == 2u &&
         work.plan.tile_count == kTileCount &&
         work.plan.dispatch_window_tiles == 4u &&
         work.plan.dispatch_count == 3u && work.artifact.ok;
}

} // namespace node_accel_contract::runtime::window::direct
