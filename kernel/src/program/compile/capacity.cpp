#include "local.hpp"
#include "../timing/local.hpp"

namespace rund::kernel::program_detail {

KernelProgramCapacityProof CheckProgramCapacity(Workspace& workspace,
                                                const WorkspaceReservation& required,
                                                const bool require_no_allocation,
                                                const char* const reason) {
  const TimePoint capacity_start = Now();
  const KernelProgramCapacityProof proof =
      BuildKernelProgramCapacityProof(workspace, required, require_no_allocation, reason);
  RecordCapacityCheckCost(workspace, capacity_start);
  return proof;
}

} // namespace rund::kernel::program_detail
