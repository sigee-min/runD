#include "local.hpp"

namespace rund::kernel::program_detail {

KernelProgramBuild
BuildInitialCapacityFailureResult(Workspace &workspace,
                                  const KernelProgramCapacityProof &proof) {
  return KernelProgramBuild{
      .ok = false,
      .reason = proof.reason,
      .program = workspace.program,
  };
}

KernelProgramBuild
BuildProgramFailureResult(Workspace &workspace, const char *const reason,
                          const PartitionBuild &schedule_build,
                          const FoldGraphBuild &fold_build) {
  return KernelProgramBuild{
      .ok = false,
      .reason = reason,
      .schedule_build = schedule_build,
      .fold_build = fold_build,
      .program = workspace.program,
  };
}

KernelProgramBuild
BuildProgramSuccessResult(Workspace &workspace,
                          const PartitionBuild &schedule_build,
                          const FoldGraphBuild &fold_build) {
  return KernelProgramBuild{
      .ok = true,
      .reason = "pass",
      .schedule_build = schedule_build,
      .fold_build = fold_build,
      .program = ViewKernelProgram(workspace),
  };
}

} // namespace rund::kernel::program_detail
