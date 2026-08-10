#include "stats.hpp"

#include <rund/counter.hpp>

#include <algorithm>

namespace rund::compute::detail {
namespace {

void add(std::uint64_t &total, const std::uint64_t value) noexcept {
  ::rund::detail::counter::Accumulate(total, value);
}

void accumulate_control(ControlStats &total,
                        const ControlStats &wave) noexcept {
  add(total.generated_item_count, wave.generated_item_count);
  add(total.generated_capacity, wave.generated_capacity);
  add(total.indirect_dispatch_count, wave.indirect_dispatch_count);
  add(total.indirect_work_item_count, wave.indirect_work_item_count);
  add(total.iteration_count, wave.iteration_count);
  add(total.skipped_iteration_count, wave.skipped_iteration_count);
  add(total.conflict_count, wave.conflict_count);
  total.overflow_ordinal =
      std::min(total.overflow_ordinal, wave.overflow_ordinal);
}

void accumulate_publication(PublicationStats &total,
                            const PublicationStats &wave) noexcept {
  total.generation = wave.generation;
  add(total.commit_count, wave.commit_count);
  add(total.discard_count, wave.discard_count);
  add(total.snapshot_byte_count, wave.snapshot_byte_count);
  total.snapshot_hash = wave.snapshot_hash;
  add(total.restore_byte_count, wave.restore_byte_count);
  add(total.device_loss_count, wave.device_loss_count);
}

void accumulate_pipeline(PipelineStats &total,
                         const PipelineStats &wave) noexcept {
  total.step_count = std::max(total.step_count, wave.step_count);
  total.resource_count = std::max(total.resource_count, wave.resource_count);
  total.barrier_count = std::max(total.barrier_count, wave.barrier_count);
  total.sealed_repetition_count =
      std::max(total.sealed_repetition_count, wave.sealed_repetition_count);
  add(total.coalesced_repetition_count, wave.coalesced_repetition_count);
  add(total.claim_conflict_count, wave.claim_conflict_count);
  add(total.verified_step_count, wave.verified_step_count);
  if (wave.failed_step_index != PipelineStats::no_failed_step) {
    total.failed_step_index = wave.failed_step_index;
  }
  add(total.status_entry_count, wave.status_entry_count);
  add(total.control_byte_count, wave.control_byte_count);
  add(total.control_command_count, wave.control_command_count);
  add(total.executed_outer_window_count, wave.executed_outer_window_count);
  add(total.skipped_outer_window_count, wave.skipped_outer_window_count);
  add(total.executed_inner_iteration_count,
      wave.executed_inner_iteration_count);
  add(total.skipped_inner_iteration_count, wave.skipped_inner_iteration_count);
  if (wave.failed_outer_window != PipelineStats::no_coordinate) {
    total.failed_outer_window = wave.failed_outer_window;
  }
  if (wave.failed_inner_iteration != PipelineStats::no_coordinate) {
    total.failed_inner_iteration = wave.failed_inner_iteration;
  }
  if (wave.failed_nested_phase != PipelineNestedPhase::None) {
    total.failed_nested_phase = wave.failed_nested_phase;
  }
  total.prepared_template_count =
      std::max(total.prepared_template_count, wave.prepared_template_count);
  total.prepared_command_count =
      std::max(total.prepared_command_count, wave.prepared_command_count);
  total.sampled_runs = std::max(total.sampled_runs, wave.sampled_runs);
  total.clean_runs = std::max(total.clean_runs, wave.clean_runs);
  add(total.claim_ns, wave.claim_ns);
  add(total.control_ns, wave.control_ns);
}

} // namespace

Status accumulate_virtual_wave(Stats &total, const Stats &wave) noexcept {
  if (!wave.available() ||
      (total.available() && total.backend != wave.backend) ||
      (total.graph_hash != 0u && wave.graph_hash != 0u &&
       total.graph_hash != wave.graph_hash)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  total.backend = wave.backend;
  add(total.pipeline_compiles, wave.pipeline_compiles);
  add(total.buffer_allocations, wave.buffer_allocations);
  add(total.download_events, wave.download_events);
  add(total.dispatches, wave.dispatches);
  add(total.command_submits, wave.command_submits);
  total.command_capacity =
      std::max(total.command_capacity, wave.command_capacity);
  total.command_inflight_peak =
      std::max(total.command_inflight_peak, wave.command_inflight_peak);
  add(total.command_capacity_rejections, wave.command_capacity_rejections);
  add(total.uploaded_bytes, wave.uploaded_bytes);
  add(total.downloaded_bytes, wave.downloaded_bytes);
  add(total.pipeline_cache_hits, wave.pipeline_cache_hits);
  add(total.pipeline_cache_evictions, wave.pipeline_cache_evictions);
  add(total.buffer_reuses, wave.buffer_reuses);
  add(total.descriptor_pool_creations, wave.descriptor_pool_creations);
  add(total.descriptor_set_allocations, wave.descriptor_set_allocations);
  add(total.descriptor_reuses, wave.descriptor_reuses);
  add(total.original_dispatches, wave.original_dispatches);
  add(total.final_dispatches, wave.final_dispatches);
  add(total.fusions, wave.fusions);
  add(total.fusion_rejections, wave.fusion_rejections);
  add(total.internal_roundtrip_bytes, wave.internal_roundtrip_bytes);
  add(total.external_roundtrip_bytes, wave.external_roundtrip_bytes);
  add(total.reset_bytes, wave.reset_bytes);
  add(total.reset_commands, wave.reset_commands);
  add(total.graph_read_bytes, wave.graph_read_bytes);
  add(total.kernel_ns, wave.kernel_ns);
  add(total.kernel_samples, wave.kernel_samples);
  add(total.shader_compile_ns, wave.shader_compile_ns);
  add(total.spirv_compile_ns, wave.spirv_compile_ns);
  add(total.pipeline_create_ns, wave.pipeline_create_ns);
  add(total.descriptor_setup_ns, wave.descriptor_setup_ns);
  add(total.submit_wait_ns, wave.submit_wait_ns);
  add(total.readback_ns, wave.readback_ns);
  if (total.graph_hash == 0u) {
    total.graph_hash = wave.graph_hash;
  }
  total.worker_count = std::max(total.worker_count, wave.worker_count);
  total.participating_workers =
      std::max(total.participating_workers, wave.participating_workers);
  add(total.tile_count, wave.tile_count);
  total.tile_size = std::max(total.tile_size, wave.tile_size);
  add(total.vector_chunks, wave.vector_chunks);
  add(total.tail_chunks, wave.tail_chunks);
  accumulate_control(total.control, wave.control);
  add(total.transfer_submissions.host_to_device,
      wave.transfer_submissions.host_to_device);
  add(total.transfer_submissions.device_to_host,
      wave.transfer_submissions.device_to_host);
  add(total.transfer_submissions.device_to_device,
      wave.transfer_submissions.device_to_device);
  add(total.host_write_bytes, wave.host_write_bytes);
  accumulate_publication(total.publication, wave.publication);
  accumulate_pipeline(total.pipeline, wave.pipeline);
  return Status::success();
}

} // namespace rund::compute::detail
