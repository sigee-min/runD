#pragma once

#include <kernel/program/executor/prepare/state.hpp>

namespace rund::kernel::executor_detail {

template <std::size_t Rank>
[[nodiscard]] inline const char *
ValidatePreparedEachProgram(const PreparedEach<Rank> &prepared) noexcept {
  if (!prepared.valid) {
    return prepared.reason;
  }
  if (prepared.exec.workspace == nullptr) {
    return "executor_workspace_missing";
  }
  const KernelProgram &program = prepared.exec.workspace->program;
  if (!program.ok) {
    return program.reason;
  }
  if (program.generation != prepared.program_generation ||
      program.schedule.packet_count != static_cast<u32>(prepared.units) ||
      program.schedule.execution_width != prepared.exec.workers ||
      program.schedule.intent != PartitionIntent::StaticWidth ||
      program.schedule.placement != PlacementPolicy::Uniform ||
      program.schedule.alignment_packets !=
          prepared.exec.boundary_alignment.units ||
      !program.schedule.no_allocation ||
      program.schedule.partition_count != prepared.partition_count) {
    return "prepared_each_program_mismatch";
  }
  return nullptr;
}

} // namespace rund::kernel::executor_detail
