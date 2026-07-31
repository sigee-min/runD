#pragma once

#include <accel/context/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/value.hpp>

#include "../evidence.hpp"
#include "../execute.hpp"
#include "dispatch.hpp"
#include "shape.hpp"

namespace rund::node::accel {

rund::AccelEvidence RunAccelKernel(const rund::AccelContext &context,
                                   const rund::AccelKernel &kernel,
                                   const rund::AccelRun &run) {
  const detail::KernelExecution execution =
      detail::AdmitKernelForExecution(context, kernel);
  if (!execution.admission.check.ok ||
      !detail::RunRequestShapeOk(context, execution, run)) {
    return detail::RejectKernelEvidence(context, execution,
                                        "accel_kernel_run_invalid");
  }

  const detail::RunBindBuild binds = detail::BuildRunBinds(execution, run);
  if (!binds.ok) {
    return detail::RejectKernelEvidence(context, execution, binds.reason);
  }
  const detail::ResetBindBuild resets =
      detail::BuildResetBinds(execution, run, binds.binds);
  if (!resets.ok) {
    return detail::RejectKernelEvidence(context, execution, resets.reason);
  }

  const detail::RunDispatchBuild dispatch =
      detail::BuildRunDispatch(execution, run);
  if (!dispatch.ok) {
    return detail::RejectKernelEvidence(context, execution, dispatch.reason);
  }

  if (run.fresh_evidence) {
    ResetRuntimeStats(context.pick);
  }
  return detail::ExecuteKernelSteps(
      context, execution, run, binds.binds, resets.binds,
      dispatch.planned_steps,
      dispatch.original_dispatch_count, dispatch.final_dispatch_count);
}

} // namespace rund::node::accel
