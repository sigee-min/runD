#pragma once

#include <kernel/program/model.hpp>
#include <kernel/program/request.hpp>
#include <kernel/reduction/fold/graph/result.hpp>
#include <kernel/schedule/planner/build.hpp>

namespace rund::kernel {

struct Workspace;

struct KernelProgramBuild {
  bool ok = false;
  const char* reason = "not_run";
  PartitionBuild schedule_build{};
  FoldGraphBuild fold_build{};
  KernelProgram program{};
};

KernelProgramBuild CompileKernelProgram(Workspace& workspace, const KernelProgramCompileRequest& request);
KernelProgram ViewKernelProgram(const Workspace& workspace);

} // namespace rund::kernel
