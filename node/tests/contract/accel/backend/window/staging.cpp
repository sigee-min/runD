#include <accel/device.hpp>

#include "local.hpp"

#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/artifact/admission.hpp>
#include <kernel/program/compute/lowering/entry.hpp>
#include <kernel/program/compute/lowering/fusion/build.hpp>
#include <kernel/program/compute/plan.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace node_accel_contract {

[[nodiscard]] bool
BackendRejectsUnderreportedStaging(const rund::AccelDevice &pick) {
  namespace fixture = backend;
  namespace admission = backend::window;
  if (!admission::CanExercise(pick)) {
    return true;
  }

  constexpr rund::kernel::u64 kMax = ~rund::kernel::u64{0u};
  const rund::kernel::u64 forged_tiles = pick.caps.max_window_tiles;
  const fixture::ComputeOpFixture op = fixture::ComputeOp(pick.caps.api);
  const rund::kernel::u64 inflated_input_bytes =
      (pick.caps.staging_bytes / forged_tiles) + 1u;
  if (inflated_input_bytes > kMax - op.map.output_bytes_per_tile ||
      inflated_input_bytes + op.map.output_bytes_per_tile >
          kMax - op.map.metadata_bytes_per_tile) {
    return true;
  }

  const rund::kernel::u64 forged_bytes_per_tile =
      inflated_input_bytes + op.map.output_bytes_per_tile +
      op.map.metadata_bytes_per_tile;
  if (forged_bytes_per_tile > kMax / forged_tiles) {
    return true;
  }

  const rund::kernel::u64 forged_tile_bytes =
      forged_bytes_per_tile * forged_tiles;
  if (forged_tile_bytes > kMax - op.map.param_bytes ||
      forged_tile_bytes + op.map.param_bytes <= pick.caps.staging_bytes) {
    return true;
  }

  std::uint32_t input_storage = 0u;
  std::vector<std::uint32_t> staged(static_cast<std::size_t>(forged_tiles));
  std::vector<rund::kernel::u64> sequence_tiles(
      static_cast<std::size_t>(forged_tiles));
  for (rund::kernel::u64 index = 0u; index < forged_tiles; ++index) {
    sequence_tiles[static_cast<std::size_t>(index)] = index;
  }

  std::array<std::uint32_t, 1u> params{7u};
  const rund::kernel::BufferSpan input_buffer{
      .data = &input_storage,
      .element_bytes = inflated_input_bytes,
      .stride_bytes = inflated_input_bytes,
      .count = forged_tiles,
  };
  rund::kernel::BindingSet bindings =
      rund::kernel::BindingSet::staged_outputs(staged.data(), staged.size());
  bindings.phase_id = fixture::Phase().phase_id;
  bindings.lane_count = 1u;
  bindings.op_hash_hi = op.map.op_hash_hi;
  bindings.op_hash_lo = op.map.op_hash_lo;
  bindings.api = pick.caps.api;
  bindings.scalar = rund::kernel::ComputeScalar::Lane32;
  bindings.input_bytes_per_tile = inflated_input_bytes;
  bindings.output_bytes_per_tile = op.map.output_bytes_per_tile;
  bindings.param_bytes = op.map.param_bytes;
  bindings.metadata_bytes_per_tile = op.map.metadata_bytes_per_tile;
  bindings.input_buffers = &input_buffer;
  bindings.input_buffer_count = op.map.input_buffer_count;
  bindings.param_data = params.data();
  bindings.param_data_bytes = sizeof(params);
  bindings.sequence_tiles = sequence_tiles.data();
  bindings.sequence_tile_count = sequence_tiles.size();

  rund::kernel::ComputePlan forged = rund::kernel::PlanCompute(
      fixture::Phase(), op.map, pick.caps, admission::Limit(pick));
  if (!forged.ok) {
    return false;
  }
  forged.tile_count = forged_tiles;
  forged.input_bytes_per_tile = inflated_input_bytes;
  forged.bytes_per_tile = forged_bytes_per_tile;
  forged.dispatch_window_tiles = forged_tiles;
  forged.dispatch_count = 1u;
  forged.staging_bytes = pick.caps.staging_bytes;

  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir, forged.api);
  const rund::kernel::ComputeDispatchWindow window =
      admission::Dispatch(forged_tiles);
  const bool accepted = pick.backend.execute(pick.backend.context, forged,
                                             artifact, &window, 1u, bindings);
  return !accepted;
}

} // namespace node_accel_contract
