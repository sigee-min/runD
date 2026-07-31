#include <accel/device.hpp>

#include "local.hpp"

#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/artifact/admission.hpp>
#include <kernel/program/compute/lowering/entry.hpp>
#include <kernel/program/compute/lowering/fusion/build.hpp>
#include <kernel/program/compute/plan.hpp>

#include <array>
#include <cstdint>

namespace node_accel_contract {

[[nodiscard]] bool
BackendRejectsPlanObligationMismatches(const rund::AccelDevice &pick) {
  namespace fixture = backend;
  namespace admission = backend::window;
  std::array<std::uint32_t, 4u> input{};
  std::array<std::uint32_t, 4u> staged{};
  std::array<std::uint32_t, 1u> params{7u};
  std::array<rund::kernel::u64, 4u> sequence_tiles{0u, 1u, 2u, 3u};
  const fixture::ComputeOpFixture op = fixture::ComputeOp(pick.caps.api);
  const rund::kernel::BufferSpan input_buffer =
      rund::kernel::BufferSpan::contiguous(input.data(), input.size());
  const rund::kernel::ComputeDispatchWindow window = admission::Dispatch(4u);

  const rund::kernel::ComputePlan plan = rund::kernel::PlanCompute(
      fixture::Phase(), op.map, pick.caps, admission::Limit(pick));
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir, plan.api);
  rund::kernel::BindingSet input_mismatch =
      rund::kernel::BindingSet::staged_outputs(staged.data(), staged.size());
  input_mismatch.phase_id = fixture::Phase().phase_id;
  input_mismatch.lane_count = 1u;
  input_mismatch.op_hash_hi = op.map.op_hash_hi;
  input_mismatch.op_hash_lo = op.map.op_hash_lo;
  input_mismatch.api = pick.caps.api;
  input_mismatch.scalar = rund::kernel::ComputeScalar::Lane32;
  input_mismatch.input_bytes_per_tile = 0u;
  input_mismatch.output_bytes_per_tile = op.map.output_bytes_per_tile;
  input_mismatch.param_bytes = op.map.param_bytes;
  input_mismatch.metadata_bytes_per_tile = op.map.metadata_bytes_per_tile;
  input_mismatch.param_data = params.data();
  input_mismatch.param_data_bytes = sizeof(params);
  input_mismatch.sequence_tiles = sequence_tiles.data();
  input_mismatch.sequence_tile_count = sequence_tiles.size();
  const bool accepted_input_mismatch =
      plan.ok && pick.backend &&
      pick.backend.execute(pick.backend.context, plan, artifact, &window, 1u,
                           input_mismatch);

  const rund::kernel::ComputePlan param_plan = rund::kernel::PlanCompute(
      fixture::Phase(), op.map, pick.caps, admission::Limit(pick));
  rund::kernel::BindingSet param_mismatch =
      rund::kernel::BindingSet::staged_outputs(staged.data(), staged.size());
  param_mismatch.phase_id = fixture::Phase().phase_id;
  param_mismatch.lane_count = 1u;
  param_mismatch.op_hash_hi = op.map.op_hash_hi;
  param_mismatch.op_hash_lo = op.map.op_hash_lo;
  param_mismatch.api = pick.caps.api;
  param_mismatch.scalar = rund::kernel::ComputeScalar::Lane32;
  param_mismatch.input_bytes_per_tile = op.map.input_bytes_per_tile;
  param_mismatch.output_bytes_per_tile = op.map.output_bytes_per_tile;
  param_mismatch.metadata_bytes_per_tile = op.map.metadata_bytes_per_tile;
  param_mismatch.input_buffers = &input_buffer;
  param_mismatch.input_buffer_count = op.map.input_buffer_count;
  param_mismatch.param_bytes = 0u;
  param_mismatch.param_data = params.data();
  param_mismatch.param_data_bytes = 0u;
  param_mismatch.sequence_tiles = sequence_tiles.data();
  param_mismatch.sequence_tile_count = sequence_tiles.size();
  const bool accepted_param_mismatch =
      param_plan.ok && pick.backend &&
      pick.backend.execute(pick.backend.context, param_plan, artifact, &window,
                           1u, param_mismatch);

  return !accepted_input_mismatch && !accepted_param_mismatch;
}

} // namespace node_accel_contract
