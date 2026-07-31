#include <kernel/program/build.hpp>

#include <kernel/reduction/fold/graph/api.hpp>
#include <kernel/schedule/workspace.hpp>

namespace rund::kernel {

KernelProgram ViewKernelProgram(const Workspace& workspace) {
  KernelProgram program = workspace.program;
  if (!program.ok) {
    return program;
  }
  const ScheduleView schedule = ViewSchedule(workspace);
  program.schedule = schedule;
  program.exec_kernel.schedule = schedule;
  program.exec_kernel.local_reduce_slot_count = schedule.fold_slot_count;
  program.fold_graph = ViewFoldGraph(workspace.fold_graph);
  program.ordered_packet_indices = schedule.ordered_packet_indices;
  program.ordered_packet_count = schedule.ordered_packet_count;
  return program;
}

} // namespace rund::kernel
