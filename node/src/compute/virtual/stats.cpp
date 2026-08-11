#include "stats.hpp"

#include <rund/counter.hpp>

#include <algorithm>

namespace rund::compute::detail {
namespace {

void add(std::uint64_t &total, const std::uint64_t value) noexcept {
  ::rund::detail::counter::Accumulate(total, value);
}

void accumulate_control(ControlStats &total,
                        const ControlStats &epoch) noexcept {
  add(total.generated_item_count, epoch.generated_item_count);
  add(total.generated_capacity, epoch.generated_capacity);
  add(total.indirect_dispatch_count, epoch.indirect_dispatch_count);
  add(total.indirect_work_item_count, epoch.indirect_work_item_count);
  add(total.iteration_count, epoch.iteration_count);
  add(total.skipped_iteration_count, epoch.skipped_iteration_count);
  add(total.conflict_count, epoch.conflict_count);
  total.overflow_ordinal =
      std::min(total.overflow_ordinal, epoch.overflow_ordinal);
}

void accumulate_publication(PublicationStats &total,
                            const PublicationStats &epoch) noexcept {
  total.generation = epoch.generation;
  add(total.commit_count, epoch.commit_count);
  add(total.discard_count, epoch.discard_count);
  add(total.snapshot_byte_count, epoch.snapshot_byte_count);
  total.snapshot_hash = epoch.snapshot_hash;
  add(total.restore_byte_count, epoch.restore_byte_count);
  add(total.device_loss_count, epoch.device_loss_count);
}

void accumulate_pipeline(PipelineStats &total,
                         const PipelineStats &epoch) noexcept {
  total.step_count = std::max(total.step_count, epoch.step_count);
  total.resource_count = std::max(total.resource_count, epoch.resource_count);
  total.barrier_count = std::max(total.barrier_count, epoch.barrier_count);
  total.sealed_repetition_count =
      std::max(total.sealed_repetition_count, epoch.sealed_repetition_count);
  add(total.coalesced_repetition_count, epoch.coalesced_repetition_count);
  add(total.claim_conflict_count, epoch.claim_conflict_count);
  add(total.verified_step_count, epoch.verified_step_count);
  if (epoch.failed_step_index != PipelineStats::no_failed_step) {
    total.failed_step_index = epoch.failed_step_index;
  }
  add(total.status_entry_count, epoch.status_entry_count);
  add(total.control_byte_count, epoch.control_byte_count);
  add(total.control_command_count, epoch.control_command_count);
  add(total.executed_outer_window_count, epoch.executed_outer_window_count);
  add(total.skipped_outer_window_count, epoch.skipped_outer_window_count);
  add(total.executed_inner_iteration_count,
      epoch.executed_inner_iteration_count);
  add(total.skipped_inner_iteration_count, epoch.skipped_inner_iteration_count);
  if (epoch.failed_outer_window != PipelineStats::no_coordinate) {
    total.failed_outer_window = epoch.failed_outer_window;
  }
  if (epoch.failed_inner_iteration != PipelineStats::no_coordinate) {
    total.failed_inner_iteration = epoch.failed_inner_iteration;
  }
  if (epoch.failed_nested_phase != PipelineNestedPhase::None) {
    total.failed_nested_phase = epoch.failed_nested_phase;
  }
  total.prepared_template_count =
      std::max(total.prepared_template_count, epoch.prepared_template_count);
  total.prepared_command_count =
      std::max(total.prepared_command_count, epoch.prepared_command_count);
  total.sampled_runs = std::max(total.sampled_runs, epoch.sampled_runs);
  total.clean_runs = std::max(total.clean_runs, epoch.clean_runs);
  add(total.claim_ns, epoch.claim_ns);
  add(total.control_ns, epoch.control_ns);
}

} // namespace

Status accumulate_virtual_epoch(Stats &total, const Stats &epoch) noexcept {
  if (!epoch.available() ||
      (total.available() && total.backend != epoch.backend) ||
      (total.graph_hash != 0u && epoch.graph_hash != 0u &&
       total.graph_hash != epoch.graph_hash)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  total.backend = epoch.backend;
  add(total.pipeline_compiles, epoch.pipeline_compiles);
  add(total.buffer_allocations, epoch.buffer_allocations);
  add(total.download_events, epoch.download_events);
  add(total.dispatches, epoch.dispatches);
  add(total.command_submits, epoch.command_submits);
  total.command_capacity =
      std::max(total.command_capacity, epoch.command_capacity);
  total.command_inflight_peak =
      std::max(total.command_inflight_peak, epoch.command_inflight_peak);
  add(total.command_capacity_rejections, epoch.command_capacity_rejections);
  add(total.uploaded_bytes, epoch.uploaded_bytes);
  add(total.downloaded_bytes, epoch.downloaded_bytes);
  add(total.pipeline_cache_hits, epoch.pipeline_cache_hits);
  add(total.pipeline_cache_evictions, epoch.pipeline_cache_evictions);
  add(total.buffer_reuses, epoch.buffer_reuses);
  add(total.descriptor_pool_creations, epoch.descriptor_pool_creations);
  add(total.descriptor_set_allocations, epoch.descriptor_set_allocations);
  add(total.descriptor_reuses, epoch.descriptor_reuses);
  add(total.original_dispatches, epoch.original_dispatches);
  add(total.final_dispatches, epoch.final_dispatches);
  add(total.fusions, epoch.fusions);
  add(total.fusion_rejections, epoch.fusion_rejections);
  add(total.internal_roundtrip_bytes, epoch.internal_roundtrip_bytes);
  add(total.external_roundtrip_bytes, epoch.external_roundtrip_bytes);
  add(total.reset_bytes, epoch.reset_bytes);
  add(total.reset_commands, epoch.reset_commands);
  add(total.graph_read_bytes, epoch.graph_read_bytes);
  add(total.kernel_ns, epoch.kernel_ns);
  add(total.kernel_samples, epoch.kernel_samples);
  add(total.shader_compile_ns, epoch.shader_compile_ns);
  add(total.spirv_compile_ns, epoch.spirv_compile_ns);
  add(total.pipeline_create_ns, epoch.pipeline_create_ns);
  add(total.descriptor_setup_ns, epoch.descriptor_setup_ns);
  add(total.submit_wait_ns, epoch.submit_wait_ns);
  add(total.readback_ns, epoch.readback_ns);
  if (total.graph_hash == 0u) {
    total.graph_hash = epoch.graph_hash;
  }
  total.worker_count = std::max(total.worker_count, epoch.worker_count);
  total.participating_workers =
      std::max(total.participating_workers, epoch.participating_workers);
  add(total.tile_count, epoch.tile_count);
  total.tile_size = std::max(total.tile_size, epoch.tile_size);
  add(total.vector_chunks, epoch.vector_chunks);
  add(total.tail_chunks, epoch.tail_chunks);
  accumulate_control(total.control, epoch.control);
  add(total.transfer_submissions.host_to_device,
      epoch.transfer_submissions.host_to_device);
  add(total.transfer_submissions.device_to_host,
      epoch.transfer_submissions.device_to_host);
  add(total.transfer_submissions.device_to_device,
      epoch.transfer_submissions.device_to_device);
  add(total.host_write_bytes, epoch.host_write_bytes);
  accumulate_publication(total.publication, epoch.publication);
  accumulate_pipeline(total.pipeline, epoch.pipeline);
  return Status::success();
}

} // namespace rund::compute::detail
