#include "report.hpp"

#include <rund/compute/stats.hpp>

#include <cstdio>

namespace rund::measure::compute::virtual_residency {
namespace {

void PrintCounter(const ::rund::compute::MemoryCounter &counter) {
  std::printf(",%llu,%llu,%llu",
              static_cast<unsigned long long>(counter.current),
              static_cast<unsigned long long>(counter.peak),
              static_cast<unsigned long long>(counter.cumulative));
}

[[nodiscard]] const char *ProcessGlobalAllocationStatus() noexcept {
  return "not_measured";
}

[[nodiscard]] const char *PreparationEvidenceStatus(
    const ::rund::compute::PreparationEvidenceSource source) noexcept {
  switch (source) {
  case ::rund::compute::PreparationEvidenceSource::OwnerLocal:
    return "owner_local_compile_cache";
  case ::rund::compute::PreparationEvidenceSource::NoNativeProducer:
    return "unavailable_no_native_producer";
  case ::rund::compute::PreparationEvidenceSource::BackendGlobalOnly:
    return "unavailable_backend_global";
  case ::rund::compute::PreparationEvidenceSource::Unavailable:
    break;
  }
  return "unavailable";
}

} // namespace

void PrintColumns() {
  std::fputs(
      "virtual_residency_columns,evidence_scope,phase,backend,status,"
      "logical_elements,active_count,slot_elements,n_over_c,plan_page_count,"
      "plan_slot_capacity,plan_logical_bytes,plan_page_bytes,"
      "plan_working_set_bytes,plan_capacity_waves,waves,dispatches,"
      "compute_submissions,h2d_submissions,"
      "d2h_submissions,d2d_submissions,uploaded_bytes,downloaded_bytes,"
      "internal_roundtrip_bytes,external_roundtrip_bytes,download_events,"
      "submit_wait_ns,readback_ns,kernel_ns,kernel_samples,backing_io_ns,"
      "loads,writebacks,"
      "sampled_runs,rund_allocation_free_runs,rund_allocation_scope,"
      "process_global_allocation_status,prepare_evidence_status,"
      "prepare_pipeline_compiles,"
      "prepare_pipeline_cache_hits,prepare_pipeline_cache_evictions,"
      "prepare_buffer_allocations,prepare_buffer_reuses,"
      "prepare_descriptor_pool_creations,prepare_descriptor_set_allocations,"
      "prepare_descriptor_reuses,prepare_shader_compile_ns,"
      "prepare_spirv_compile_ns,prepare_pipeline_create_ns,"
      "prepare_descriptor_setup_ns,terminal_pipeline_compiles,"
      "terminal_pipeline_cache_hits,terminal_pipeline_cache_evictions,"
      "terminal_buffer_allocations,terminal_buffer_reuses,"
      "terminal_descriptor_pool_creations,"
      "terminal_descriptor_set_allocations,terminal_descriptor_reuses,"
      "plan_peak_bytes,plan_committed_peak_bytes,"
      "plan_scratch_payload_bytes,plan_scratch_bytes,"
      "memory_host_current_bytes,memory_host_peak_bytes,"
      "memory_host_cumulative_bytes,memory_device_current_bytes,"
      "memory_device_peak_bytes,memory_device_cumulative_bytes,"
      "memory_resident_current_bytes,memory_resident_peak_bytes,"
      "memory_resident_cumulative_bytes,memory_staging_current_bytes,"
      "memory_staging_peak_bytes,memory_staging_cumulative_bytes,"
      "memory_transfer_current_bytes,memory_transfer_peak_bytes,"
      "memory_transfer_cumulative_bytes,plan_hash_hi,plan_hash_lo,graph_hash,"
      "result_hash,backing_input_bytes,backing_output_bytes,terminal_reads,"
      "profile_projections,first_result_us,warm_p50_us,warm_p95_us,"
      "logical_elements_per_s,author_us,compile_us,prepare_us,seed_us,run_us,"
      "read_us,first_value_bits\n",
      stdout);
}

void PrintEvidence(const char *const phase, const Backend backend,
                   const std::size_t active_count, const WallEvidence &wall,
                   const std::uint32_t first_value_bits,
                   const ::rund::compute::telemetry::Profile &profile,
                   const ::rund::compute::PipelinePlan &plan,
                   const ::rund::compute::telemetry::Profile &preparation,
                   const ObservationEvidence &observation) {
  const ::rund::compute::Stats &stats = profile.execution();
  const ::rund::compute::MemoryStats &memory = profile.memory();
  const ::rund::compute::Stats &preparation_stats = preparation.execution();
  const auto &residency = stats.pipeline.residency;
  std::printf("virtual_residency,current_source_diagnostic,%s,%s,ok,%zu,%zu,"
              "%zu,%.9f,%llu,%llu,%llu,%llu,%llu,%llu",
              phase, Name(backend), LogicalElements, active_count, SlotElements,
              static_cast<double>(active_count) /
                  static_cast<double>(SlotElements),
              static_cast<unsigned long long>(plan.residency.page_count),
              static_cast<unsigned long long>(plan.residency.slot_capacity),
              static_cast<unsigned long long>(plan.residency.logical_bytes),
              static_cast<unsigned long long>(plan.residency.page_bytes),
              static_cast<unsigned long long>(plan.residency.working_set_bytes),
              static_cast<unsigned long long>(plan.residency.wave_count));
  std::printf(",%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
              "%llu,%llu,%llu,%llu",
              static_cast<unsigned long long>(residency.wave_count),
              static_cast<unsigned long long>(stats.dispatches),
              static_cast<unsigned long long>(stats.command_submits),
              static_cast<unsigned long long>(
                  stats.transfer_submissions.host_to_device),
              static_cast<unsigned long long>(
                  stats.transfer_submissions.device_to_host),
              static_cast<unsigned long long>(
                  stats.transfer_submissions.device_to_device),
              static_cast<unsigned long long>(stats.uploaded_bytes),
              static_cast<unsigned long long>(stats.downloaded_bytes),
              static_cast<unsigned long long>(stats.internal_roundtrip_bytes),
              static_cast<unsigned long long>(stats.external_roundtrip_bytes),
              static_cast<unsigned long long>(stats.download_events),
              static_cast<unsigned long long>(stats.submit_wait_ns),
              static_cast<unsigned long long>(stats.readback_ns),
              static_cast<unsigned long long>(stats.kernel_ns),
              static_cast<unsigned long long>(stats.kernel_samples),
              static_cast<unsigned long long>(residency.backing_io_ns));
  std::printf(",%llu,%llu,%u,%u,rund_owned,%s,%s",
              static_cast<unsigned long long>(residency.load_count),
              static_cast<unsigned long long>(residency.writeback_count),
              residency.sampled_runs, residency.allocation_free_runs,
              ProcessGlobalAllocationStatus(),
              PreparationEvidenceStatus(
                  preparation_stats.pipeline.preparation_evidence));
  std::printf(
      ",%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu",
      static_cast<unsigned long long>(preparation_stats.pipeline_compiles),
      static_cast<unsigned long long>(preparation_stats.pipeline_cache_hits),
      static_cast<unsigned long long>(
          preparation_stats.pipeline_cache_evictions),
      static_cast<unsigned long long>(preparation_stats.buffer_allocations),
      static_cast<unsigned long long>(preparation_stats.buffer_reuses),
      static_cast<unsigned long long>(
          preparation_stats.descriptor_pool_creations),
      static_cast<unsigned long long>(
          preparation_stats.descriptor_set_allocations),
      static_cast<unsigned long long>(preparation_stats.descriptor_reuses),
      static_cast<unsigned long long>(preparation_stats.shader_compile_ns),
      static_cast<unsigned long long>(preparation_stats.spirv_compile_ns),
      static_cast<unsigned long long>(preparation_stats.pipeline_create_ns),
      static_cast<unsigned long long>(preparation_stats.descriptor_setup_ns));
  std::printf(",%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu",
              static_cast<unsigned long long>(stats.pipeline_compiles),
              static_cast<unsigned long long>(stats.pipeline_cache_hits),
              static_cast<unsigned long long>(stats.pipeline_cache_evictions),
              static_cast<unsigned long long>(stats.buffer_allocations),
              static_cast<unsigned long long>(stats.buffer_reuses),
              static_cast<unsigned long long>(stats.descriptor_pool_creations),
              static_cast<unsigned long long>(stats.descriptor_set_allocations),
              static_cast<unsigned long long>(stats.descriptor_reuses));
  std::printf(",%llu,%llu,%llu,%llu",
              static_cast<unsigned long long>(plan.peak_bytes),
              static_cast<unsigned long long>(plan.committed_peak_bytes),
              static_cast<unsigned long long>(plan.scratch_payload_bytes),
              static_cast<unsigned long long>(plan.scratch_bytes));
  PrintCounter(memory.host);
  PrintCounter(memory.device);
  PrintCounter(memory.resident);
  PrintCounter(memory.staging);
  PrintCounter(memory.transfer);
  std::printf(",%llu,%llu,%llu,%llu,%llu,%llu,%u,%u",
              static_cast<unsigned long long>(residency.plan_identity_hi),
              static_cast<unsigned long long>(residency.plan_identity_lo),
              static_cast<unsigned long long>(stats.graph_hash),
              static_cast<unsigned long long>(stats.output_hash),
              static_cast<unsigned long long>(residency.backing_read_bytes),
              static_cast<unsigned long long>(residency.backing_write_bytes),
              observation.terminal_reads, observation.profile_projections);
  std::printf(",%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%u\n",
              wall.first_result_us, wall.warm_p50_us, wall.warm_p95_us,
              wall.logical_elements_per_s, wall.author_us, wall.compile_us,
              wall.prepare_us, wall.seed_us, wall.run_us, wall.read_us,
              first_value_bits);
}

} // namespace rund::measure::compute::virtual_residency
