#include "internal.hpp"

#include "../../job/state.hpp"

#include <memory>
#include <type_traits>

namespace rund::compute::detail {

bool CpuPreparedArena::materialize(const CpuPreparedArenaPlan &plan) noexcept {
  static_assert(
      std::is_nothrow_default_constructible_v<kernel::ComputeTileRunStorage>);
  static_assert(std::is_nothrow_destructible_v<kernel::ComputeTileRunStorage>);
  static_assert(std::is_nothrow_default_constructible_v<CpuMapRoute>);
  static_assert(std::is_nothrow_destructible_v<CpuMapRoute>);
  static_assert(std::is_nothrow_default_constructible_v<
                node::accel::cpu_simd_detail::CpuSimdReadBinding>);
  static_assert(std::is_nothrow_destructible_v<
                node::accel::cpu_simd_detail::CpuSimdReadBinding>);
  static_assert(std::is_nothrow_default_constructible_v<
                node::accel::cpu_simd_detail::CpuSimdWriteBinding>);
  static_assert(std::is_nothrow_destructible_v<
                node::accel::cpu_simd_detail::CpuSimdWriteBinding>);
  static_assert(
      std::is_nothrow_default_constructible_v<std::shared_ptr<BufferState>>);
  static_assert(std::is_nothrow_destructible_v<std::shared_ptr<BufferState>>);
  static_assert(std::is_nothrow_default_constructible_v<JobBufferView>);
  static_assert(std::is_nothrow_destructible_v<JobBufferView>);
  static_assert(std::is_nothrow_default_constructible_v<
                node::accel::detail::KernelViewSlot>);
  static_assert(
      std::is_nothrow_destructible_v<node::accel::detail::KernelViewSlot>);
  static_assert(std::is_nothrow_default_constructible_v<CpuViewTransfer>);
  static_assert(std::is_nothrow_destructible_v<CpuViewTransfer>);
  static_assert(std::is_nothrow_default_constructible_v<JobWorkspace>);
  static_assert(std::is_nothrow_destructible_v<JobWorkspace>);

  const CpuExecutionStoragePlan &execution = plan.execution;
  if (!plan.layout.sealed || mapping_.committed_bytes() != 0u ||
      (execution.tiles.ok &&
       !cpu_prepared_detail::valid_tile_plan(execution.tiles)) ||
      !mapping_.allocate(plan.layout)) {
    return false;
  }
  tile_state_ =
      mapping_.construct<kernel::ComputeTileRunStorage>(plan.tile_state);
  failure_slots_ = mapping_.construct<const char *>(plan.failure_slots);
  worker_tiles_ = mapping_.construct<kernel::u32>(plan.worker_tiles);
  worker_stats_partitions_ =
      mapping_.construct<kernel::u32>(plan.worker_stats_partitions);
  worker_stats_start_offset_ns_ =
      mapping_.construct<kernel::u64>(plan.worker_stats_start_offset_ns);
  worker_stats_elapsed_ns_ =
      mapping_.construct<kernel::u64>(plan.worker_stats_elapsed_ns);
  worker_stats_tail_wait_ns_ =
      mapping_.construct<kernel::u64>(plan.worker_stats_tail_wait_ns);
  map_scratch_ = mapping_.construct<std::max_align_t>(plan.map_scratch);
  simd_ = mapping_.construct<CpuSimdCount>(plan.simd);
  collective_totals_ =
      mapping_.construct<CpuCollectiveWide>(plan.collective_totals);
  collective_prefixes_ =
      mapping_.construct<CpuCollectiveWide>(plan.collective_prefixes);
  primitive_u32_ = mapping_.construct<kernel::u32>(plan.primitive_u32);
  primitive_u64_ = mapping_.construct<kernel::u64>(plan.primitive_u64);
  primitive_i32_ = mapping_.construct<kernel::i32>(plan.primitive_i32);
  primitive_i64_ = mapping_.construct<kernel::i64>(plan.primitive_i64);
  scatter_keys_ = mapping_.construct<kernel::u32>(plan.scatter_keys);
  scatter_marks_ = mapping_.construct<kernel::u32>(plan.scatter_marks);
  scatter_epoch_ = mapping_.construct<kernel::u32>(plan.scatter_epoch);
  transform_i32_ = mapping_.construct<kernel::i32>(plan.transform_i32);
  transform_i64_ = mapping_.construct<kernel::i64>(plan.transform_i64);
  primitive_objects_ = mapping_.bytes(plan.primitive_objects);
  maps_ = mapping_.construct<CpuMapRoute>(plan.maps);
  reads_ = mapping_.construct<node::accel::cpu_simd_detail::CpuSimdReadBinding>(
      plan.reads);
  writes_ =
      mapping_.construct<node::accel::cpu_simd_detail::CpuSimdWriteBinding>(
          plan.writes);
  buffer_owners_ =
      mapping_.construct<std::shared_ptr<BufferState>>(plan.buffer_owners);
  buffer_views_ = mapping_.construct<JobBufferView>(plan.buffer_views);
  kernel_views_ = mapping_.construct<node::accel::detail::KernelViewSlot>(
      plan.kernel_views);
  view_transfers_ = mapping_.construct<CpuViewTransfer>(plan.view_transfers);
  workspace_offsets_ = mapping_.construct<std::size_t>(plan.workspace_offsets);
  workspaces_ = mapping_.construct<JobWorkspace>(plan.workspaces);
  const bool complete =
      tile_state_.size() == (execution.tiles.ok ? 1u : 0u) &&
      failure_slots_.size() == execution.tiles.failure_slot_capacity &&
      worker_tiles_.size() == execution.tiles.worker_capacity &&
      worker_stats_partitions_.size() == execution.tiles.worker_capacity &&
      worker_stats_start_offset_ns_.size() == execution.tiles.worker_capacity &&
      worker_stats_elapsed_ns_.size() == execution.tiles.worker_capacity &&
      worker_stats_tail_wait_ns_.size() == execution.tiles.worker_capacity &&
      map_scratch_.size() == execution.map_scratch_count &&
      simd_.size() == execution.simd_count &&
      collective_totals_.size() == execution.collective_total_count &&
      collective_prefixes_.size() == execution.collective_prefix_count &&
      primitive_u32_.size() == execution.primitive_u32_count &&
      primitive_u64_.size() == execution.primitive_u64_count &&
      primitive_i32_.size() == execution.primitive_i32_count &&
      primitive_i64_.size() == execution.primitive_i64_count &&
      scatter_keys_.size() == execution.scatter_slot_count &&
      scatter_marks_.size() == execution.scatter_slot_count &&
      scatter_epoch_.size() == (execution.scatter_slot_count == 0u ? 0u : 1u) &&
      transform_i32_.size() == execution.transform_i32_count &&
      transform_i64_.size() == execution.transform_i64_count &&
      primitive_objects_.size() == execution.primitive_object_storage_bytes &&
      maps_.size() == plan.map_count && reads_.size() == plan.read_count &&
      writes_.size() == plan.write_count &&
      buffer_owners_.size() == plan.buffer_owner_count &&
      buffer_views_.size() == plan.buffer_view_count &&
      kernel_views_.size() == plan.kernel_view_count &&
      view_transfers_.size() == plan.view_transfer_count &&
      workspaces_.size() == plan.workspace_count &&
      workspace_offsets_.size() == plan.workspace_offset_count;
  if (!complete) {
    clear();
    return false;
  }
  tile_capacity_ = execution.tiles;
  const CpuStorageBytes payload = cpu_prepared_arena_payload(plan);
  payload_host_bytes_ = payload.host;
  payload_tile_bytes_ = payload.tile;
  return true;
}

} // namespace rund::compute::detail
