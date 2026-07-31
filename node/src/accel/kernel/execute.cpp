#include <accel/check.hpp>
#include <accel/context/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/runtime.hpp>
#include <rund/counter.hpp>

#include "../backend/ops/table.hpp"
#include "../backend/resource.hpp"
#include "backend/run.hpp"
#include "evidence.hpp"
#include "execute.hpp"
#include "roundtrip.hpp"
#include "schedule.hpp"
#include <cstddef>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] rund::AccelEvidence EvidenceFromCheck(
    const rund::AccelContext &context, const KernelExecution &execution,
    const std::uint64_t original_dispatch_count,
    const std::uint64_t final_dispatch_count, const rund::AccelCheck &check) {
  const rund::RuntimeStats stats =
      ReadBackendStats(execution.context_admission.pick);
  return EvidenceFromStats(context, execution, stats, original_dispatch_count,
                           final_dispatch_count, false, check.reason, 0u, 0u,
                           check.failed_batches, check.first_failed_batch,
                           check.first_status);
}

} // namespace

rund::AccelEvidence
ExecuteKernelSteps(const rund::AccelContext &context,
                   const KernelExecution &execution, const rund::AccelRun &run,
                   const RunBinds &run_binds,
                   const BoundResets &resets,
                   const PlannedStepStorage &planned_steps,
                   const std::uint64_t original_dispatch_count,
                   const std::uint64_t final_dispatch_count) {
  const ScheduledStepOrder step_order = BuildScheduledStepOrder(
      execution.steps, execution.graph_roles,
      execution.graph_alias_representatives, execution.required_barriers);
  if (!step_order.ok || step_order.size() != execution.steps.size()) {
    return RejectKernelEvidence(context, execution, "accel_kernel_run_invalid");
  }
  const ProducerConsumerRoundtrip roundtrip =
      CountProducerConsumerRoundtripBytes(execution, step_order, run);
  if (!roundtrip.ok) {
    return RejectKernelEvidence(context, execution, roundtrip.reason);
  }

  BoundRun bound =
      BuildBoundRun(context, execution, run_binds, resets, planned_steps,
                    step_order, original_dispatch_count, final_dispatch_count);
  bound.run.resets = &resets;
  if (!bound.ok || bound.run.ops == nullptr) {
    return RejectKernelEvidence(context, execution, "accel_kernel_run_invalid");
  }
  if (bound.run.ops->run == nullptr) {
    return RejectKernelEvidence(context, execution, "accel_kernel_run_invalid");
  }
  std::uint64_t backend_traffic = 0u;
  bound.run.traffic = &backend_traffic;
  const rund::AccelCheck finish = bound.run.ops->run(bound.run);
  if (!finish.ok) {
    return EvidenceFromCheck(context, execution, original_dispatch_count,
                             final_dispatch_count, finish);
  }

  const rund::RuntimeStats stats =
      ReadBackendStats(execution.context_admission.pick);
  if (!stats.ok) {
    return EvidenceFromStats(context, execution, stats, original_dispatch_count,
                             final_dispatch_count, false, stats.reason);
  }
  const std::uint64_t physical_dispatch_count =
      stats.dispatch_count == 0u ? final_dispatch_count : stats.dispatch_count;
  const std::uint64_t internal_bytes =
      ::rund::detail::counter::SaturatingAdd(roundtrip.internal_bytes,
                                             backend_traffic);
  return EvidenceFromStats(context, execution, stats, original_dispatch_count,
                           physical_dispatch_count, true, "ok",
                           internal_bytes, roundtrip.external_bytes,
                           finish.failed_batches, finish.first_failed_batch,
                           finish.first_status);
}

} // namespace rund::node::accel::detail
