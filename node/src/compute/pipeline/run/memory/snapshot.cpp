#include "internal.hpp"

#include "../../local.hpp"
#include "../../state.hpp"

#include "../../../memory/local.hpp"

#include <mutex>

namespace rund::compute::detail {

MemorySnapshot
pipeline_memory_snapshot(const std::shared_ptr<PipelineState> &state,
                         const std::span<MemoryEntry> entries) noexcept {
  if (!valid_pipeline(state)) {
    SnapshotWriter writer{MemoryStats{}, entries};
    return writer.finish();
  }
  std::lock_guard lock{state->gate};
  const PipelineMemoryView view = pipeline_memory_view_locked(*state);
  const MemoryStats &summary = view.summary;
  SnapshotWriter writer{summary, entries};
  const std::uint64_t metadata = view.metadata;
  writer.add(MemoryCategory::Host, MemoryUse::Metadata, 0u,
             fixed_memory(metadata));
  if (state->device->backend == Backend::Cpu &&
      summary.host.current > metadata) {
    writer.add(MemoryCategory::Host, MemoryUse::Internal, 0u,
               fixed_memory(::rund::detail::counter::Remaining(
                                summary.host.current, metadata),
                            summary.host.reused));
  } else if (state->device->backend != Backend::Cpu) {
    const MemoryCounter native_host =
        pipeline_prepared_memory(view.prepared.host);
    if (native_host.current != 0u || native_host.cumulative != 0u) {
      writer.add(MemoryCategory::Host, MemoryUse::Metadata, 1u, native_host);
    }
  }
  if (summary.resident.current != 0u) {
    const MemoryCounter scratch = fixed_memory(view.scratch.resident);
    const MemoryCounter internal =
        pipeline_remaining_memory(summary.resident, scratch);
    if (internal.current != 0u || internal.cumulative != 0u) {
      writer.add(MemoryCategory::Resident, MemoryUse::Internal, 0u, internal);
    }
    if (scratch.current != 0u) {
      writer.add(MemoryCategory::Resident, MemoryUse::Scratch, 0u, scratch);
    }
  }
  if (summary.device.current != 0u) {
    const MemoryCounter scratch =
        fixed_memory(view.scratch.physical, view.scratch.reused);
    const MemoryCounter internal =
        pipeline_remaining_memory(summary.device, scratch);
    if (internal.current != 0u || internal.cumulative != 0u) {
      writer.add(MemoryCategory::Device, MemoryUse::Internal, 0u, internal);
    }
    if (scratch.current != 0u) {
      writer.add(MemoryCategory::Device, MemoryUse::Scratch, 0u, scratch);
    }
  }
  if (summary.tile.current != 0u) {
    writer.add(MemoryCategory::Tile, MemoryUse::Scratch, 0u, summary.tile);
  }
  if (summary.staging.current != 0u) {
    writer.add(MemoryCategory::Staging, MemoryUse::Coordinator, 0u,
               summary.staging);
  }
  if (summary.frame.current != 0u || summary.frame.cumulative != 0u) {
    writer.add(MemoryCategory::Frame, MemoryUse::Coordinator, 0u,
               summary.frame);
  }
  if (summary.transfer.peak != 0u || summary.transfer.cumulative != 0u) {
    writer.add(MemoryCategory::Transfer, MemoryUse::Traffic, 0u,
               summary.transfer);
  }
  return writer.finish();
}

} // namespace rund::compute::detail
