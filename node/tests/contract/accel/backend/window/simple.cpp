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

[[nodiscard]] bool BackendAcceptsSimpleWindow(const rund::AccelDevice &pick) {
  namespace fixture = backend;
  namespace admission = backend::window;
  std::array<std::uint32_t, 4u> input{};
  std::array<std::uint32_t, 4u> staged{};
  std::array<std::uint32_t, 1u> params{7u};
  std::array<rund::kernel::u64, 4u> sequence_tiles{0u, 1u, 2u, 3u};
  const fixture::ComputeOpFixture op = fixture::ComputeOp(pick.caps.api);
  const rund::kernel::BufferSpan input_buffer =
      rund::kernel::BufferSpan::contiguous(input.data(), input.size());
  rund::kernel::BindingSet bindings =
      rund::kernel::BindingSet::staged_outputs(staged.data(), staged.size());
  bindings.phase_id = fixture::Phase().phase_id;
  bindings.lane_count = 1u;
  bindings.op_hash_hi = op.map.op_hash_hi;
  bindings.op_hash_lo = op.map.op_hash_lo;
  bindings.api = pick.caps.api;
  bindings.scalar = rund::kernel::ComputeScalar::Lane32;
  bindings.input_bytes_per_tile = op.map.input_bytes_per_tile;
  bindings.output_bytes_per_tile = op.map.output_bytes_per_tile;
  bindings.param_bytes = op.map.param_bytes;
  bindings.metadata_bytes_per_tile = op.map.metadata_bytes_per_tile;
  bindings.input_buffers = &input_buffer;
  bindings.input_buffer_count = op.map.input_buffer_count;
  bindings.param_data = params.data();
  bindings.param_data_bytes = sizeof(params);
  bindings.sequence_tiles = sequence_tiles.data();
  bindings.sequence_tile_count = sequence_tiles.size();

  const rund::kernel::ComputePlan plan = rund::kernel::PlanCompute(
      fixture::Phase(), op.map, pick.caps, admission::Limit(pick));
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir, plan.api);
  const rund::kernel::ComputeDispatchWindow window = admission::Dispatch(4u);

  return plan.ok && pick.backend &&
         pick.backend.execute(pick.backend.context, plan, artifact, &window, 1u,
                              bindings);
}

} // namespace node_accel_contract
