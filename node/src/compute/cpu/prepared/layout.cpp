#include "internal.hpp"

#include "../../job/state.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::compute::detail {

bool seal_cpu_prepared_arena_plan(CpuPreparedArenaPlan &plan,
                                  const std::uint64_t page_bytes) noexcept {
  const CpuExecutionStoragePlan &execution = plan.execution;
  if (plan.layout.sealed ||
      (execution.tiles.ok &&
       !cpu_prepared_detail::valid_tile_plan(execution.tiles)) ||
      (!execution.tiles.ok &&
       (execution.map_scratch_count != 0u || execution.simd_count != 0u ||
        execution.collective_total_count != 0u ||
        execution.collective_prefix_count != 0u)) ||
      execution.collective_prefix_count > execution.collective_total_count ||
      execution.primitive_object_payload_bytes >
          execution.primitive_object_storage_bytes ||
      (execution.primitive_object_storage_bytes == 0u) !=
          (execution.primitive_object_payload_bytes == 0u) ||
      execution.primitive_object_storage_bytes % alignof(std::max_align_t) !=
          0u ||
      !append_cpu_arena_segment<kernel::ComputeTileRunStorage>(
          plan.layout, execution.tiles.ok ? 1u : 0u, plan.tile_state) ||
      !append_cpu_arena_segment<const char *>(
          plan.layout, execution.tiles.failure_slot_capacity,
          plan.failure_slots) ||
      !append_cpu_arena_segment<kernel::u32>(
          plan.layout, execution.tiles.worker_capacity, plan.worker_tiles) ||
      !append_cpu_arena_segment<kernel::u32>(plan.layout,
                                             execution.tiles.worker_capacity,
                                             plan.worker_stats_partitions) ||
      !append_cpu_arena_segment<kernel::u64>(
          plan.layout, execution.tiles.worker_capacity,
          plan.worker_stats_start_offset_ns) ||
      !append_cpu_arena_segment<kernel::u64>(plan.layout,
                                             execution.tiles.worker_capacity,
                                             plan.worker_stats_elapsed_ns) ||
      !append_cpu_arena_segment<kernel::u64>(plan.layout,
                                             execution.tiles.worker_capacity,
                                             plan.worker_stats_tail_wait_ns) ||
      !append_cpu_arena_segment(
          plan.layout, execution.map_scratch_count, sizeof(std::max_align_t),
          kCpuWorkerWriteIsolationBytes, plan.map_scratch) ||
      !append_cpu_arena_segment<CpuSimdCount>(plan.layout, execution.simd_count,
                                              plan.simd) ||
      !append_cpu_arena_segment<CpuCollectiveWide>(
          plan.layout, execution.collective_total_count,
          plan.collective_totals) ||
      !append_cpu_arena_segment<CpuCollectiveWide>(
          plan.layout, execution.collective_prefix_count,
          plan.collective_prefixes) ||
      !append_cpu_arena_segment<kernel::u32>(
          plan.layout, execution.primitive_u32_count, plan.primitive_u32) ||
      !append_cpu_arena_segment<kernel::u64>(
          plan.layout, execution.primitive_u64_count, plan.primitive_u64) ||
      !append_cpu_arena_segment<kernel::i32>(
          plan.layout, execution.primitive_i32_count, plan.primitive_i32) ||
      !append_cpu_arena_segment<kernel::i64>(
          plan.layout, execution.primitive_i64_count, plan.primitive_i64) ||
      !append_cpu_arena_segment<kernel::u32>(
          plan.layout, execution.scatter_slot_count, plan.scatter_keys) ||
      !append_cpu_arena_segment<kernel::u32>(
          plan.layout, execution.scatter_slot_count, plan.scatter_marks) ||
      !append_cpu_arena_segment<kernel::u32>(
          plan.layout, execution.scatter_slot_count == 0u ? 0u : 1u,
          plan.scatter_epoch) ||
      !append_cpu_arena_segment<kernel::i32>(
          plan.layout, execution.transform_i32_count, plan.transform_i32) ||
      !append_cpu_arena_segment<kernel::i64>(
          plan.layout, execution.transform_i64_count, plan.transform_i64) ||
      !append_cpu_arena_segment(
          plan.layout, execution.primitive_object_storage_bytes, 1u,
          alignof(std::max_align_t), plan.primitive_objects) ||
      !append_cpu_arena_segment<CpuMapRoute>(plan.layout, plan.map_count,
                                             plan.maps) ||
      !append_cpu_arena_segment<
          node::accel::cpu_simd_detail::CpuSimdReadBinding>(
          plan.layout, plan.read_count, plan.reads) ||
      !append_cpu_arena_segment<
          node::accel::cpu_simd_detail::CpuSimdWriteBinding>(
          plan.layout, plan.write_count, plan.writes) ||
      !append_cpu_arena_segment<std::shared_ptr<BufferState>>(
          plan.layout, plan.buffer_owner_count, plan.buffer_owners) ||
      !append_cpu_arena_segment<JobBufferView>(
          plan.layout, plan.buffer_view_count, plan.buffer_views) ||
      !append_cpu_arena_segment<node::accel::detail::KernelViewSlot>(
          plan.layout, plan.kernel_view_count, plan.kernel_views) ||
      !append_cpu_arena_segment<CpuViewTransfer>(
          plan.layout, plan.view_transfer_count, plan.view_transfers) ||
      !append_cpu_arena_segment<std::size_t>(
          plan.layout, plan.workspace_offset_count, plan.workspace_offsets) ||
      !append_cpu_arena_segment<JobWorkspace>(plan.layout, plan.workspace_count,
                                              plan.workspaces)) {
    return false;
  }
  return seal_cpu_arena_layout(plan.layout, page_bytes);
}

CpuStorageBytes
cpu_execution_storage_payload(const CpuExecutionStoragePlan &plan) noexcept {
  using cpu_prepared_detail::saturating_add;
  using cpu_prepared_detail::saturating_extent;
  if (!cpu_prepared_detail::has_execution(plan) ||
      (plan.tiles.ok && !cpu_prepared_detail::valid_tile_plan(plan.tiles))) {
    return {};
  }
  const kernel::ComputeTileRetainedMemory tiles =
      plan.tiles.ok ? plan.tiles.retained.memory
                    : kernel::ComputeTileRetainedMemory{};
  std::uint64_t tile = 0u;
  tile = saturating_add(tile, tiles.workspace_bytes);
  tile = saturating_add(tile, tiles.failure_slot_bytes);
  tile = saturating_add(tile, tiles.worker_tile_bytes);
  tile = saturating_add(tile, tiles.async_context_bytes);
  tile = saturating_add(tile, saturating_extent(plan.map_scratch_count,
                                                sizeof(std::max_align_t)));
  tile = saturating_add(
      tile, saturating_extent(plan.simd_count, sizeof(CpuSimdCount)));
  tile = saturating_add(tile, saturating_extent(plan.collective_total_count,
                                                sizeof(CpuCollectiveWide)));
  tile = saturating_add(tile, saturating_extent(plan.collective_prefix_count,
                                                sizeof(CpuCollectiveWide)));
  tile = saturating_add(
      tile, saturating_extent(plan.primitive_u32_count, sizeof(kernel::u32)));
  tile = saturating_add(
      tile, saturating_extent(plan.primitive_u64_count, sizeof(kernel::u64)));
  tile = saturating_add(
      tile, saturating_extent(plan.primitive_i32_count, sizeof(kernel::i32)));
  tile = saturating_add(
      tile, saturating_extent(plan.primitive_i64_count, sizeof(kernel::i64)));
  tile = saturating_add(
      tile, saturating_extent(plan.scatter_slot_count, sizeof(kernel::u32)));
  tile = saturating_add(
      tile, saturating_extent(plan.scatter_slot_count, sizeof(kernel::u32)));
  tile = saturating_add(
      tile, saturating_extent(plan.scatter_slot_count == 0u ? 0u : 1u,
                              sizeof(kernel::u32)));
  tile = saturating_add(
      tile, saturating_extent(plan.transform_i32_count, sizeof(kernel::i32)));
  tile = saturating_add(
      tile, saturating_extent(plan.transform_i64_count, sizeof(kernel::i64)));
  return CpuStorageBytes{
      .host = saturating_add(tiles.state_bytes,
                             plan.primitive_object_payload_bytes),
      .tile = tile,
  };
}

CpuStorageBytes
cpu_prepared_arena_payload(const CpuPreparedArenaPlan &plan) noexcept {
  CpuStorageBytes payload = cpu_execution_storage_payload(plan.execution);
  payload.host =
      cpu_prepared_detail::saturating_add(payload.host, plan.maps.size_bytes);
  payload.host =
      cpu_prepared_detail::saturating_add(payload.host, plan.reads.size_bytes);
  payload.host =
      cpu_prepared_detail::saturating_add(payload.host, plan.writes.size_bytes);
  payload.host = cpu_prepared_detail::saturating_add(
      payload.host, plan.buffer_owners.size_bytes);
  payload.host = cpu_prepared_detail::saturating_add(
      payload.host, plan.buffer_views.size_bytes);
  payload.host = cpu_prepared_detail::saturating_add(
      payload.host, plan.kernel_views.size_bytes);
  payload.host = cpu_prepared_detail::saturating_add(
      payload.host, plan.view_transfers.size_bytes);
  payload.host = cpu_prepared_detail::saturating_add(
      payload.host, plan.workspaces.size_bytes);
  payload.host = cpu_prepared_detail::saturating_add(
      payload.host, plan.workspace_offsets.size_bytes);
  return payload;
}

} // namespace rund::compute::detail
