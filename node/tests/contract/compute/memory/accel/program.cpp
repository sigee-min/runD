#include "../model.hpp"

#include "../../../target/selection.hpp"

#include <node/runtime/compute/access.hpp>

#include "../../../../../src/accel/graph/token.hpp"
#include "../../../../../src/compute/flow/state.hpp"
#include "../../../../../src/compute/program/state.hpp"

#include <optional>

namespace rund_node_memory_contract {

int CheckAccelProgramHostAccounting(const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto device = open(rund::node::test_contract::target_for(backend));
  if (!device) {
    return 1;
  }
  auto program =
      on(*device)
          .input<std::uint32_t>(32u)
          .map("accel-owner-map", [](auto value) { return value + 1u; })
          .scan(Scan::InclusiveSum)
          .compile();
  if (!program) {
    return 2;
  }
  const auto &state = detail::FlowAccess::state(*program);
  if (state == nullptr || state->accel == nullptr ||
      state->cpu_graph != nullptr ||
      state->accel->kernel_token_host_bytes == 0u ||
      state->chunks.size() != 1u || state->graph_bindings.empty() ||
      state->graph_info.nodes.size() != 2u) {
    return 3;
  }
  const std::optional<std::uint64_t> measured =
      rund::node::accel::detail::MeasureKernelTokenRetainedMemory(
          state->accel->kernel);
  rund::AccelKernel tampered = state->accel->kernel;
  ++tampered.node_count;
  if (!measured.has_value() ||
      *measured != state->accel->kernel_token_host_bytes ||
      rund::node::accel::detail::MeasureKernelTokenRetainedMemory(tampered)
          .has_value()) {
    return 4;
  }
  MemoryStats memory{};
  if (!ReadMemory(*program, memory)) {
    return 5;
  }
  MemoryStats snapshot_stats{};
  const SnapshotAccounting snapshot = SnapshotMemory(*program, snapshot_stats);
  const PhysicalInternal physical = PhysicalInternalMemory(*program);
  const std::uint64_t owner_floor = sizeof(detail::ProgramState) +
                                    sizeof(detail::AccelProgram) +
                                    state->accel->kernel_token_host_bytes;
  if (!snapshot.complete || !snapshot.valid || !snapshot.allocation_free ||
      snapshot.metadata_entries != 1u || snapshot.tile_entries != 1u ||
      snapshot.internal_entries != 0u || !SameStats(memory, snapshot_stats) ||
      !physical.complete || physical.count != 0u || physical.bytes != 0u ||
      memory.host.current < owner_floor || memory.device.current != 0u ||
      memory.tile.current != 0u) {
    return 6;
  }
  return 0;
}

} // namespace rund_node_memory_contract
