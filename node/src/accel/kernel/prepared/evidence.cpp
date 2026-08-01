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
  return stats.dispatch_count == 0u ? planned : stats.dispatch_count;
}

[[nodiscard]] rund::AccelCheck
PipelineOutcome(const PipelineState &pipeline,
                const KernelResult &backend) noexcept {
  if (!backend.stats.ok) {
    return rund::AccelCheck{false, backend.stats.reason};
  }
  if (!backend.check.ok) {
    return backend.check;
  }
  if (!backend.pipeline.submitted && !backend.pipeline.control_observed &&
      backend.stats.dispatch_count == 0u &&
      backend.stats.command_submit_count == 0u) {
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
  stats.dispatch_count = check.ok ? final_dispatches : 0u;
  return rund::AccelEvidence{
      .backend = context.api,
      .dispatch_count = stats.dispatch_count,
      .command_submit_count = stats.command_submit_count,
      .command_capacity = stats.command_capacity,
      .command_inflight_peak = stats.command_inflight_peak,
      .command_capacity_rejection_count =
          stats.command_capacity_rejection_count,
      .reset_command_count = stats.reset_command_count,
      .reset_bytes = stats.reset_bytes,
      .original_operation_count = counts.original_operations,
      .fused_operation_count = counts.fused_operations,
      .original_dispatch_count = counts.original_dispatches,
      .final_dispatch_count = final_dispatches,
      .fusion_rejection_count = counts.fusion_rejections,
      .fusion_reason = counts.fusion_rejections == 0u ? "ok" : "batch",
      .internal_producer_consumer_roundtrip_bytes =
          check.ok ? counts.internal_roundtrip_bytes : 0u,
      .external_producer_consumer_roundtrip_bytes =
          check.ok ? counts.external_roundtrip_bytes : 0u,
      .host_to_device_bytes = stats.host_to_device_bytes,
      .device_to_host_bytes = stats.device_to_host_bytes,
      .pipeline_compile_count = stats.pipeline_compile_count,
      .pipeline_cache_hit_count = stats.pipeline_cache_hit_count,
      .pipeline_cache_eviction_count = stats.pipeline_cache_eviction_count,
      .descriptor_pool_create_count = stats.descriptor_pool_create_count,
      .descriptor_set_allocate_count = stats.descriptor_set_allocate_count,
      .buffer_reuse_hit_count = stats.buffer_reuse_hit_count,
      .buffer_allocation_count = stats.buffer_allocation_count,
      .descriptor_reuse_hit_count = stats.descriptor_reuse_hit_count,
      .accel_kernel_ns = stats.accel_kernel_ns,
      .accel_timestamp_count = stats.accel_timestamp_count,
      .accel_timestamp_source = stats.accel_timestamp_source,
      .shader_compile_ns = stats.shader_compile_ns,
      .spirv_compile_ns = stats.spirv_compile_ns,
      .pipeline_create_ns = stats.pipeline_create_ns,
      .descriptor_setup_ns = stats.descriptor_setup_ns,
      .command_submit_wait_ns = stats.command_submit_wait_ns,
      .readback_ns = stats.readback_ns,
      .generated_item_count = stats.generated_item_count,
      .generated_capacity = stats.generated_capacity,
      .indirect_dispatch_count = stats.indirect_dispatch_count,
      .indirect_work_item_count = stats.indirect_work_item_count,
      .iteration_count = stats.iteration_count,
      .skipped_iteration_count = stats.skipped_iteration_count,
      .conflict_count = stats.conflict_count,
      .overflow_ordinal = stats.overflow_ordinal,
      .ok = check.ok && stats.ok,
      .reason = !check.ok ? check.reason : stats.reason,
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
  if (!stats.ok) {
    return EvidenceFromStats(
        context, state.execution, stats, state.dispatch.original_dispatch_count,
        state.dispatch.final_dispatch_count, false, stats.reason);
  }
  return EvidenceFromStats(
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
