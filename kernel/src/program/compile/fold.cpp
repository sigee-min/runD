#include "local.hpp"
#include "../timing/local.hpp"

#include <kernel/reduction/fold/graph/api.hpp>

namespace rund::kernel::program_detail {

FoldGraphBuild CompileProgramFoldGraph(Workspace& workspace,
                                       const PartitionBuild& schedule_build,
                                       const KernelProgramCompileRequest& request,
                                       const AllocationPolicy allocation) {
  const TimePoint fold_graph_start = Now();
  const FoldGraphBuild fold_build =
      BuildFoldGraph(workspace.fold_graph,
                     schedule_build.fold_slot_count,
                     request.fold_operation,
                     request.strict_float_reduction,
                     allocation);
  RecordFoldGraphCompileCost(workspace, fold_graph_start);
  return fold_build;
}

} // namespace rund::kernel::program_detail
