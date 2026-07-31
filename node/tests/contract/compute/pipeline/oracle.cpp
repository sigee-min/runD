#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/program/state.hpp"

#include <limits>

namespace rund_node_test_pipeline {

[[nodiscard]] bool CanonicalChunkOrder(
    const rund::compute::detail::ProgramState &program) noexcept {
  if (program.chunk_order.size() != program.chunks.size()) {
    return false;
  }
  for (std::size_t rank = 0u; rank < program.chunk_order.size(); ++rank) {
    const std::size_t index = program.chunk_order[rank];
    if (index >= program.chunks.size()) {
      return false;
    }
    for (std::size_t prior = 0u; prior < rank; ++prior) {
      if (program.chunk_order[prior] == index) {
        return false;
      }
    }
    if (rank == 0u) {
      continue;
    }
    const std::size_t previous = program.chunk_order[rank - 1u];
    const std::size_t previous_count = program.chunks[previous].count;
    const std::size_t count = program.chunks[index].count;
    if (previous_count < count ||
        (previous_count == count && previous >= index)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool WarmCountersClean(const Stats &stats) noexcept {
  return stats.pipeline_compiles == 0u && stats.buffer_allocations == 0u &&
         stats.pipeline_cache_evictions == 0u &&
         stats.descriptor_pool_creations == 0u &&
         stats.descriptor_set_allocations == 0u && stats.uploaded_bytes == 0u &&
         stats.download_events == 0u && stats.downloaded_bytes == 0u &&
         stats.internal_roundtrip_bytes == 0u &&
         stats.external_roundtrip_bytes == 0u && stats.output_hash == 0u;
}

[[nodiscard]] bool
SameMemoryCounter(const rund::compute::MemoryCounter &left,
                  const rund::compute::MemoryCounter &right) noexcept {
  return left.current == right.current && left.peak == right.peak &&
         left.cumulative == right.cumulative && left.reused == right.reused &&
         left.budget == right.budget;
}

[[nodiscard]] bool
SameMemory(const rund::compute::MemoryStats &left,
           const rund::compute::MemoryStats &right) noexcept {
  return left.backend == right.backend && left.scope == right.scope &&
         SameMemoryCounter(left.host, right.host) &&
         SameMemoryCounter(left.frame, right.frame) &&
         SameMemoryCounter(left.tile, right.tile) &&
         SameMemoryCounter(left.resident, right.resident) &&
         SameMemoryCounter(left.staging, right.staging) &&
         SameMemoryCounter(left.device, right.device) &&
         SameMemoryCounter(left.transfer, right.transfer);
}

void AddMemoryCounter(rund::compute::MemoryCounter &total,
                      const rund::compute::MemoryCounter &part) noexcept {
  const auto add = [](std::uint64_t &target,
                      const std::uint64_t value) noexcept {
    constexpr std::uint64_t limit = std::numeric_limits<std::uint64_t>::max();
    target = value > limit - target ? limit : target + value;
  };
  add(total.current, part.current);
  add(total.peak, part.peak);
  add(total.cumulative, part.cumulative);
  add(total.reused, part.reused);
  add(total.budget, part.budget);
}

void AddMemory(rund::compute::MemoryStats &total,
               const rund::compute::MemoryStats &part) noexcept {
  AddMemoryCounter(total.host, part.host);
  AddMemoryCounter(total.frame, part.frame);
  AddMemoryCounter(total.tile, part.tile);
  AddMemoryCounter(total.resident, part.resident);
  AddMemoryCounter(total.staging, part.staging);
  AddMemoryCounter(total.device, part.device);
  AddMemoryCounter(total.transfer, part.transfer);
}

[[nodiscard]] bool ProfileMemoryReconciles(
    const rund::compute::PipelineProfileSnapshot &snapshot,
    const std::span<const rund::compute::PipelineStepProfile> steps) noexcept {
  rund::compute::MemoryStats reconstructed = snapshot.shared_memory;
  for (const rund::compute::PipelineStepProfile &step : steps) {
    AddMemory(reconstructed, step.memory);
  }
  return SameMemory(reconstructed, snapshot.memory);
}

[[nodiscard]] bool
TimingUnavailable(const rund::compute::StepTiming &timing) noexcept {
  return !timing.available() && timing.duration_ns == 0u &&
         timing.sample_count == 0u &&
         timing.clock == rund::compute::StepClock::Unavailable &&
         timing.relation == rund::compute::StepTimingRelation::Unavailable;
}

[[nodiscard]] bool
SameControlStats(const rund::compute::ControlStats &left,
                 const rund::compute::ControlStats &right) noexcept {
  return left.generated_item_count == right.generated_item_count &&
         left.generated_capacity == right.generated_capacity &&
         left.indirect_dispatch_count == right.indirect_dispatch_count &&
         left.indirect_work_item_count == right.indirect_work_item_count &&
         left.iteration_count == right.iteration_count &&
         left.skipped_iteration_count == right.skipped_iteration_count &&
         left.conflict_count == right.conflict_count &&
         left.overflow_ordinal == right.overflow_ordinal;
}

[[nodiscard]] bool SameWarmStats(const rund::compute::Stats &left,
                                 const rund::compute::Stats &right) noexcept {
  const auto same_publication =
      right.publication.generation == left.publication.generation + 1u &&
      left.publication.commit_count == right.publication.commit_count &&
      left.publication.discard_count == right.publication.discard_count &&
      left.publication.snapshot_byte_count ==
          right.publication.snapshot_byte_count &&
      left.publication.snapshot_hash == right.publication.snapshot_hash &&
      left.publication.restore_byte_count ==
          right.publication.restore_byte_count &&
      left.publication.device_loss_count == right.publication.device_loss_count;
  const auto same_pipeline =
      left.pipeline.step_count == right.pipeline.step_count &&
      left.pipeline.resource_count == right.pipeline.resource_count &&
      left.pipeline.barrier_count == right.pipeline.barrier_count &&
      left.pipeline.claim_conflict_count ==
          right.pipeline.claim_conflict_count &&
      left.pipeline.verified_step_count == right.pipeline.verified_step_count &&
      left.pipeline.failed_step_index == right.pipeline.failed_step_index &&
      left.pipeline.status_entry_count == right.pipeline.status_entry_count &&
      left.pipeline.control_byte_count == right.pipeline.control_byte_count &&
      left.pipeline.control_command_count ==
          right.pipeline.control_command_count;
  return left.backend == right.backend &&
         left.pipeline_compiles == right.pipeline_compiles &&
         left.buffer_allocations == right.buffer_allocations &&
         left.download_events == right.download_events &&
         left.dispatches == right.dispatches &&
         left.command_submits == right.command_submits &&
         left.command_capacity == right.command_capacity &&
         left.command_inflight_peak == right.command_inflight_peak &&
         left.command_capacity_rejections ==
             right.command_capacity_rejections &&
         left.uploaded_bytes == right.uploaded_bytes &&
         left.downloaded_bytes == right.downloaded_bytes &&
         left.pipeline_cache_hits == right.pipeline_cache_hits &&
         left.pipeline_cache_evictions == right.pipeline_cache_evictions &&
         left.buffer_reuses == right.buffer_reuses &&
         left.descriptor_pool_creations == right.descriptor_pool_creations &&
         left.descriptor_set_allocations == right.descriptor_set_allocations &&
         left.descriptor_reuses == right.descriptor_reuses &&
         left.original_dispatches == right.original_dispatches &&
         left.final_dispatches == right.final_dispatches &&
         left.fusions == right.fusions &&
         left.fusion_rejections == right.fusion_rejections &&
         left.internal_roundtrip_bytes == right.internal_roundtrip_bytes &&
         left.external_roundtrip_bytes == right.external_roundtrip_bytes &&
         left.graph_read_bytes == right.graph_read_bytes &&
         left.graph_hash == right.graph_hash &&
         left.output_hash == right.output_hash &&
         left.worker_count == right.worker_count &&
         left.participating_workers == right.participating_workers &&
         left.tile_count == right.tile_count &&
         left.tile_size == right.tile_size &&
         left.vector_chunks == right.vector_chunks &&
         left.tail_chunks == right.tail_chunks &&
         SameControlStats(left.control, right.control) && same_publication &&
         same_pipeline;
}

[[nodiscard]] bool SameMemoryEntries(
    const std::span<const rund::compute::MemoryEntry> left,
    const std::span<const rund::compute::MemoryEntry> right) noexcept {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < left.size(); ++index) {
    if (left[index].category != right[index].category ||
        left[index].use != right[index].use ||
        left[index].index != right[index].index ||
        !SameMemoryCounter(left[index].bytes, right[index].bytes)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool Overwrite(Buffer<std::uint32_t> &buffer,
                             const std::span<const std::uint32_t> values) {
  rund::compute::WriteStats stats{};
  return static_cast<bool>(rund::compute::detail::write_buffer(
      rund::compute::detail::BufferAccess::state(buffer),
      rund::compute::detail::HostView{
          values.data(), values.size(),
          rund::compute::detail::type<std::uint32_t>()},
      stats));
}

}  // namespace rund_node_test_pipeline
