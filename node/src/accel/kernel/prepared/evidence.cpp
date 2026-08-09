#include "evidence.hpp"

#include "../evidence.hpp"

#include <rund/counter.hpp>

namespace rund::node::accel::detail::prepared {
namespace {

[[nodiscard]] std::uint64_t
PhysicalDispatchCount(const rund::RuntimeStats &stats,
                      const std::uint64_t planned) noexcept {
  // CPU and empty backends report no physical counter. Accelerator backends
  // may append physical gather/scatter work to the authored dispatch plan.
  return stats.run.work.dispatch_count == 0u ? planned
                                             : stats.run.work.dispatch_count;
}

[[nodiscard]] rund::AccelCheck
PipelineOutcome(const PipelineState &pipeline,
                const KernelResult &backend) noexcept {
  if (!backend.stats.outcome.ok) {
    return rund::AccelCheck{false, backend.stats.outcome.reason};
  }
  if (!backend.check.ok) {
    return backend.check;
  }
  if (!backend.pipeline.submitted && !backend.pipeline.control_observed &&
      backend.stats.run.work.dispatch_count == 0u &&
      backend.stats.run.work.command_submit_count == 0u) {
    return rund::AccelCheck{true, "ok"};
  }
  if (!backend.pipeline.control_observed ||
      !ValidPreparedPipelineControl(backend.pipeline.control,
                                    pipeline.status)) {
    return rund::AccelCheck{false, "compute_backend_failed"};
  }
  return backend.pipeline.control.reason ==
                 static_cast<std::uint32_t>(rund::compute::Reason::Ok)
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{
                   false, CanonicalReasonText(backend.pipeline.control.reason)};
}

} // namespace

void Accumulate(EvidenceCounts &counts, const RunState &state) noexcept {
  Accumulate(counts, state, 1u);
}

void Accumulate(EvidenceCounts &counts, const RunState &state,
                const std::uint64_t occurrences) noexcept {
  const auto scale = [occurrences](const std::uint64_t value) noexcept {
    return ::rund::detail::counter::SaturatingMultiply(value, occurrences);
  };
  counts.original_operations = ::rund::detail::counter::SaturatingAdd(
      counts.original_operations,
      scale(state.execution.original_operation_count));
  counts.fused_operations = ::rund::detail::counter::SaturatingAdd(
      counts.fused_operations, scale(state.execution.fused_operation_count));
  counts.original_dispatches = ::rund::detail::counter::SaturatingAdd(
      counts.original_dispatches,
      scale(state.dispatch.original_dispatch_count));
  counts.final_dispatches = ::rund::detail::counter::SaturatingAdd(
      counts.final_dispatches, scale(state.dispatch.final_dispatch_count));
  counts.fusion_rejections = ::rund::detail::counter::SaturatingAdd(
      counts.fusion_rejections, scale(state.execution.fusion_rejection_count));
  counts.internal_roundtrip_bytes = ::rund::detail::counter::SaturatingAdd(
      counts.internal_roundtrip_bytes, scale(state.roundtrip.internal_bytes));
  counts.external_roundtrip_bytes = ::rund::detail::counter::SaturatingAdd(
      counts.external_roundtrip_bytes, scale(state.roundtrip.external_bytes));
}

rund::AccelEvidence BatchEvidence(const rund::AccelContext &context,
                                  rund::RuntimeStats stats,
                                  const EvidenceCounts &counts,
                                  const rund::AccelCheck check) noexcept {
  const std::uint64_t final_dispatches =
      check.ok ? PhysicalDispatchCount(stats, counts.final_dispatches)
               : counts.final_dispatches;
  stats.run.work.dispatch_count = check.ok ? final_dispatches : 0u;
  stats.run.work.original_operation_count = counts.original_operations;
  stats.run.work.fused_operation_count = counts.fused_operations;
  stats.run.work.original_dispatch_count = counts.original_dispatches;
  stats.run.work.final_dispatch_count = final_dispatches;
  stats.run.work.fusion_rejection_count = counts.fusion_rejections;
  stats.run.work.fusion_reason =
      counts.fusion_rejections == 0u ? "ok" : "batch";
  stats.run.transfer.internal_producer_consumer_roundtrip_bytes =
      check.ok ? counts.internal_roundtrip_bytes : 0u;
  stats.run.transfer.external_producer_consumer_roundtrip_bytes =
      check.ok ? counts.external_roundtrip_bytes : 0u;
  return rund::AccelEvidence{
      .identity = {.backend = context.api},
      .run = stats.run,
      .outcome =
          {
              .ok = check.ok && stats.outcome.ok,
              .reason = !check.ok ? check.reason : stats.outcome.reason,
          },
  };
}

rund::AccelEvidence RunEvidence(const rund::AccelContext &context,
                                const RunState &state,
                                const KernelResult &run) {
  const rund::RuntimeStats &stats = run.stats;
  const rund::AccelCheck &check = run.check;
  const std::uint64_t final_dispatch_count =
      check.ok
          ? PhysicalDispatchCount(stats, state.dispatch.final_dispatch_count)
          : state.dispatch.final_dispatch_count;
  if (!stats.outcome.ok) {
    return BuildKernelEvidence(
        context, state.execution, stats, state.dispatch.original_dispatch_count,
        state.dispatch.final_dispatch_count, false, stats.outcome.reason);
  }
  return BuildKernelEvidence(
      context, state.execution, stats, state.dispatch.original_dispatch_count,
      final_dispatch_count, check.ok, check.reason,
      check.ok ? state.roundtrip.internal_bytes : 0u,
      check.ok ? state.roundtrip.external_bytes : 0u, check.failed_batches,
      check.first_failed_batch, check.first_status);
}

PreparedPipelineEvidence
PipelineEvidence(const rund::AccelContext &context,
                 const PipelineState &pipeline,
                 const KernelResult &backend) noexcept {
  const rund::AccelCheck overall = PipelineOutcome(pipeline, backend);
  const bool control_valid =
      backend.pipeline.control_observed &&
      ValidPreparedPipelineControl(backend.pipeline.control, pipeline.status);
  return PreparedPipelineEvidence{
      .shared = BatchEvidence(context, backend.stats, pipeline.counts, overall),
      .check = overall,
      .control = backend.pipeline.control_observed ? backend.pipeline.control
                                                   : PreparedPipelineControl{},
      .profile = backend.pipeline.profile,
      .status_entry_count = pipeline.status.status_entry_count,
      .control_byte_count =
          backend.pipeline.control_observed ? PreparedPipelineControlBytes : 0u,
      .control_command_count = backend.pipeline.control_command_count,
      .control_ns = backend.pipeline.control_ns,
      .active_step_count = pipeline.status.active_step_count,
      .submitted = backend.pipeline.submitted,
      .control_observed = backend.pipeline.control_observed,
      .control_valid = control_valid,
  };
}

} // namespace rund::node::accel::detail::prepared
