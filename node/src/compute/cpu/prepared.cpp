#include "prepared.hpp"

#include "../job/state.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool
same_tile_plan(const kernel::ComputeTileRunStoragePlan &left,
               const kernel::ComputeTileRunStoragePlan &right) noexcept {
  const auto &a = left.retained.memory;
  const auto &b = right.retained.memory;
  return left.failure_slot_capacity == right.failure_slot_capacity &&
         left.worker_capacity == right.worker_capacity && left.ok == right.ok &&
         left.retained.ok == right.retained.ok &&
         a.state_bytes == b.state_bytes &&
         a.workspace_bytes == b.workspace_bytes &&
         a.failure_slot_bytes == b.failure_slot_bytes &&
         a.worker_tile_bytes == b.worker_tile_bytes &&
         a.async_context_bytes == b.async_context_bytes &&
         a.total_bytes == b.total_bytes;
}

[[nodiscard]] bool
valid_tile_plan(const kernel::ComputeTileRunStoragePlan &plan) noexcept {
  if (!plan.ok || !plan.retained.ok) {
    return false;
  }
  return same_tile_plan(
      plan, kernel::PlanComputeTileRunStorage(plan.failure_slot_capacity,
                                              plan.worker_capacity));
}

[[nodiscard]] bool has_execution(const CpuExecutionStoragePlan &plan) noexcept {
  return plan.tiles.ok || plan.map_scratch_count != 0u ||
         plan.simd_count != 0u || plan.collective_total_count != 0u ||
         plan.collective_prefix_count != 0u || plan.primitive_u32_count != 0u ||
         plan.primitive_u64_count != 0u || plan.primitive_i32_count != 0u ||
         plan.primitive_i64_count != 0u || plan.scatter_slot_count != 0u ||
         plan.transform_i32_count != 0u || plan.transform_i64_count != 0u ||
         plan.primitive_object_storage_bytes != 0u;
}

[[nodiscard]] std::uint64_t extent(const std::size_t count,
                                   const std::size_t width) noexcept {
  std::uint64_t result = 0u;
  return kernel::checked::mul(static_cast<std::uint64_t>(count),
                              static_cast<std::uint64_t>(width), result)
             ? result
             : std::numeric_limits<std::uint64_t>::max();
}

[[nodiscard]] std::uint64_t add(const std::uint64_t left,
                                const std::uint64_t right) noexcept {
  std::uint64_t result = 0u;
  return kernel::checked::add(left, right, result)
             ? result
             : std::numeric_limits<std::uint64_t>::max();
}

} // namespace

bool cpu_execution_storage_required(
    const CpuExecutionStoragePlan &plan) noexcept {
  return has_execution(plan);
}

bool merge_cpu_execution_storage_plan(
    CpuExecutionStoragePlan &target,
    const CpuExecutionStoragePlan &source) noexcept {
  if ((source.tiles.ok && !valid_tile_plan(source.tiles)) ||
      (!source.tiles.ok &&
       (source.map_scratch_count != 0u || source.simd_count != 0u ||
        source.collective_total_count != 0u ||
        source.collective_prefix_count != 0u)) ||
      source.primitive_object_payload_bytes >
          source.primitive_object_storage_bytes ||
      (source.primitive_object_storage_bytes == 0u) !=
          (source.primitive_object_payload_bytes == 0u) ||
      source.primitive_object_storage_bytes % alignof(std::max_align_t) != 0u) {
    return false;
  }
  CpuExecutionStoragePlan next = target;
  if (source.tiles.ok) {
    next.tiles =
        next.tiles.ok
            ? kernel::MergeComputeTileRunStoragePlans(next.tiles, source.tiles)
            : source.tiles;
    if (!valid_tile_plan(next.tiles)) {
      return false;
    }
  }
  next.map_scratch_count =
      std::max(next.map_scratch_count, source.map_scratch_count);
  next.simd_count = std::max(next.simd_count, source.simd_count);
  next.collective_total_count =
      std::max(next.collective_total_count, source.collective_total_count);
  next.collective_prefix_count =
      std::max(next.collective_prefix_count, source.collective_prefix_count);
  next.primitive_u32_count =
      std::max(next.primitive_u32_count, source.primitive_u32_count);
  next.primitive_u64_count =
      std::max(next.primitive_u64_count, source.primitive_u64_count);
  next.primitive_i32_count =
      std::max(next.primitive_i32_count, source.primitive_i32_count);
  next.primitive_i64_count =
      std::max(next.primitive_i64_count, source.primitive_i64_count);
  next.scatter_slot_count =
      std::max(next.scatter_slot_count, source.scatter_slot_count);
  if (source.transform_i32_count >
          std::numeric_limits<std::size_t>::max() - next.transform_i32_count ||
      source.transform_i64_count >
          std::numeric_limits<std::size_t>::max() - next.transform_i64_count ||
      source.primitive_object_storage_bytes >
          std::numeric_limits<std::size_t>::max() -
              next.primitive_object_storage_bytes ||
      source.primitive_object_payload_bytes >
          std::numeric_limits<std::size_t>::max() -
              next.primitive_object_payload_bytes) {
    return false;
  }
  next.transform_i32_count += source.transform_i32_count;
  next.transform_i64_count += source.transform_i64_count;
  next.primitive_object_storage_bytes += source.primitive_object_storage_bytes;
  next.primitive_object_payload_bytes += source.primitive_object_payload_bytes;
  target = next;
  return true;
}

bool seal_cpu_prepared_arena_plan(CpuPreparedArenaPlan &plan,
                                  const std::uint64_t page_bytes) noexcept {
  const CpuExecutionStoragePlan &execution = plan.execution;
  if (plan.layout.sealed ||
      (execution.tiles.ok && !valid_tile_plan(execution.tiles)) ||
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
      !append_cpu_arena_segment<std::max_align_t>(
          plan.layout, execution.map_scratch_count, plan.map_scratch) ||
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
  if (!has_execution(plan) || (plan.tiles.ok && !valid_tile_plan(plan.tiles))) {
    return {};
  }
  const kernel::ComputeTileRetainedMemory tiles =
      plan.tiles.ok ? plan.tiles.retained.memory
                    : kernel::ComputeTileRetainedMemory{};
  std::uint64_t tile = 0u;
  tile = add(tile, tiles.workspace_bytes);
  tile = add(tile, tiles.failure_slot_bytes);
  tile = add(tile, tiles.worker_tile_bytes);
  tile = add(tile, tiles.async_context_bytes);
  tile = add(tile, extent(plan.map_scratch_count, sizeof(std::max_align_t)));
  tile = add(tile, extent(plan.simd_count, sizeof(CpuSimdCount)));
  tile =
      add(tile, extent(plan.collective_total_count, sizeof(CpuCollectiveWide)));
  tile = add(tile,
             extent(plan.collective_prefix_count, sizeof(CpuCollectiveWide)));
  tile = add(tile, extent(plan.primitive_u32_count, sizeof(kernel::u32)));
  tile = add(tile, extent(plan.primitive_u64_count, sizeof(kernel::u64)));
  tile = add(tile, extent(plan.primitive_i32_count, sizeof(kernel::i32)));
  tile = add(tile, extent(plan.primitive_i64_count, sizeof(kernel::i64)));
  tile = add(tile, extent(plan.scatter_slot_count, sizeof(kernel::u32)));
  tile = add(tile, extent(plan.scatter_slot_count, sizeof(kernel::u32)));
  tile = add(tile, extent(plan.scatter_slot_count == 0u ? 0u : 1u,
                          sizeof(kernel::u32)));
  tile = add(tile, extent(plan.transform_i32_count, sizeof(kernel::i32)));
  tile = add(tile, extent(plan.transform_i64_count, sizeof(kernel::i64)));
  return CpuStorageBytes{
      .host = add(tiles.state_bytes, plan.primitive_object_payload_bytes),
      .tile = tile,
  };
}

CpuStorageBytes
cpu_prepared_arena_payload(const CpuPreparedArenaPlan &plan) noexcept {
  CpuStorageBytes payload = cpu_execution_storage_payload(plan.execution);
  payload.host = add(payload.host, plan.maps.size_bytes);
  payload.host = add(payload.host, plan.reads.size_bytes);
  payload.host = add(payload.host, plan.writes.size_bytes);
  payload.host = add(payload.host, plan.buffer_owners.size_bytes);
  payload.host = add(payload.host, plan.buffer_views.size_bytes);
  payload.host = add(payload.host, plan.kernel_views.size_bytes);
  payload.host = add(payload.host, plan.view_transfers.size_bytes);
  payload.host = add(payload.host, plan.workspaces.size_bytes);
  payload.host = add(payload.host, plan.workspace_offsets.size_bytes);
  return payload;
}

void CpuPreparedArena::clear() noexcept {
  mapping_.destroy(workspaces_);
  mapping_.destroy(workspace_offsets_);
  mapping_.destroy(view_transfers_);
  mapping_.destroy(kernel_views_);
  mapping_.destroy(buffer_views_);
  mapping_.destroy(buffer_owners_);
  mapping_.destroy(writes_);
  mapping_.destroy(reads_);
  mapping_.destroy(maps_);
  mapping_.destroy(transform_i64_);
  mapping_.destroy(transform_i32_);
  mapping_.destroy(scatter_epoch_);
  mapping_.destroy(scatter_marks_);
  mapping_.destroy(scatter_keys_);
  mapping_.destroy(primitive_i64_);
  mapping_.destroy(primitive_i32_);
  mapping_.destroy(primitive_u64_);
  mapping_.destroy(primitive_u32_);
  mapping_.destroy(collective_prefixes_);
  mapping_.destroy(collective_totals_);
  mapping_.destroy(simd_);
  mapping_.destroy(map_scratch_);
  mapping_.destroy(worker_stats_tail_wait_ns_);
  mapping_.destroy(worker_stats_elapsed_ns_);
  mapping_.destroy(worker_stats_start_offset_ns_);
  mapping_.destroy(worker_stats_partitions_);
  mapping_.destroy(worker_tiles_);
  mapping_.destroy(failure_slots_);
  mapping_.destroy(tile_state_);

  workspace_offsets_ = {};
  workspaces_ = {};
  view_transfers_ = {};
  kernel_views_ = {};
  buffer_views_ = {};
  buffer_owners_ = {};
  writes_ = {};
  reads_ = {};
  maps_ = {};
  transform_i64_ = {};
  transform_i32_ = {};
  primitive_objects_ = {};
  primitive_i64_ = {};
  primitive_i32_ = {};
  scatter_epoch_ = {};
  scatter_marks_ = {};
  scatter_keys_ = {};
  primitive_u64_ = {};
  primitive_u32_ = {};
  collective_prefixes_ = {};
  collective_totals_ = {};
  simd_ = {};
  map_scratch_ = {};
  worker_stats_tail_wait_ns_ = {};
  worker_stats_elapsed_ns_ = {};
  worker_stats_start_offset_ns_ = {};
  worker_stats_partitions_ = {};
  worker_tiles_ = {};
  failure_slots_ = {};
  tile_state_ = {};
  tile_capacity_ = {};
  transform_i32_claimed_ = 0u;
  transform_i64_claimed_ = 0u;
  primitive_objects_claimed_ = 0u;
  payload_host_bytes_ = 0u;
  payload_tile_bytes_ = 0u;
  mapping_.release();
}

CpuPreparedArena::~CpuPreparedArena() { clear(); }

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
      (execution.tiles.ok && !valid_tile_plan(execution.tiles)) ||
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

bool CpuPreparedArena::supports(
    const CpuExecutionStoragePlan &plan) const noexcept {
  const bool tiles_fit =
      !plan.tiles.ok ||
      (valid_tile_plan(plan.tiles) && valid_tile_plan(tile_capacity_) &&
       plan.tiles.failure_slot_capacity <= failure_slots_.size() &&
       plan.tiles.worker_capacity <= worker_tiles_.size() &&
       plan.tiles.worker_capacity <= worker_stats_partitions_.size() &&
       plan.tiles.worker_capacity <= worker_stats_start_offset_ns_.size() &&
       plan.tiles.worker_capacity <= worker_stats_elapsed_ns_.size() &&
       plan.tiles.worker_capacity <= worker_stats_tail_wait_ns_.size());
  return tiles_fit && plan.map_scratch_count <= map_scratch_.size() &&
         plan.simd_count <= simd_.size() &&
         plan.collective_total_count <= collective_totals_.size() &&
         plan.collective_prefix_count <= collective_prefixes_.size() &&
         plan.primitive_u32_count <= primitive_u32_.size() &&
         plan.primitive_u64_count <= primitive_u64_.size() &&
         plan.primitive_i32_count <= primitive_i32_.size() &&
         plan.primitive_i64_count <= primitive_i64_.size() &&
         plan.scatter_slot_count <= scatter_keys_.size() &&
         plan.scatter_slot_count <= scatter_marks_.size() &&
         (plan.scatter_slot_count == 0u || scatter_epoch_.size() == 1u) &&
         plan.transform_i32_count <= transform_i32_.size() &&
         plan.transform_i64_count <= transform_i64_.size() &&
         plan.primitive_object_storage_bytes <= primitive_objects_.size();
}

kernel::ComputeTileRunStorageView CpuPreparedArena::tile_storage() noexcept {
  return kernel::ComputeTileRunStorageView{
      .state = tile_state_.size() == 1u ? tile_state_.data() : nullptr,
      .failure_slots = failure_slots_,
      .worker_tiles = worker_tiles_,
      .worker_stats_partitions = worker_stats_partitions_,
      .worker_stats_start_offset_ns = worker_stats_start_offset_ns_,
      .worker_stats_elapsed_ns = worker_stats_elapsed_ns_,
      .worker_stats_tail_wait_ns = worker_stats_tail_wait_ns_,
  };
}

std::span<kernel::i32>
CpuPreparedArena::claim_transform_i32(const std::size_t count) noexcept {
  if (transform_i32_claimed_ > transform_i32_.size() ||
      count > transform_i32_.size() - transform_i32_claimed_) {
    return {};
  }
  const std::span result =
      transform_i32_.subspan(transform_i32_claimed_, count);
  transform_i32_claimed_ += count;
  return result;
}

std::span<kernel::i64>
CpuPreparedArena::claim_transform_i64(const std::size_t count) noexcept {
  if (transform_i64_claimed_ > transform_i64_.size() ||
      count > transform_i64_.size() - transform_i64_claimed_) {
    return {};
  }
  const std::span result =
      transform_i64_.subspan(transform_i64_claimed_, count);
  transform_i64_claimed_ += count;
  return result;
}

bool CpuPreparedArena::view(
    const CpuRunRouteSlice &slice, std::span<CpuMapRoute> &maps,
    std::span<node::accel::cpu_simd_detail::CpuSimdReadBinding> &reads,
    std::span<node::accel::cpu_simd_detail::CpuSimdWriteBinding>
        &writes) noexcept {
  if (slice.map_begin > maps_.size() ||
      slice.map_count > maps_.size() - slice.map_begin ||
      slice.read_begin > reads_.size() ||
      slice.read_count > reads_.size() - slice.read_begin ||
      slice.write_begin > writes_.size() ||
      slice.write_count > writes_.size() - slice.write_begin) {
    return false;
  }
  maps = maps_.subspan(slice.map_begin, slice.map_count);
  reads = reads_.subspan(slice.read_begin, slice.read_count);
  writes = writes_.subspan(slice.write_begin, slice.write_count);
  return true;
}

bool CpuPreparedArena::view(const CpuJobBindingSlice &slice,
                            CpuJobBindingStorage &storage) noexcept {
  const auto within = [](const std::size_t begin, const std::size_t count,
                         const std::size_t capacity) noexcept {
    return begin <= capacity && count <= capacity - begin;
  };
  if (!within(slice.input_begin, slice.input_count, buffer_owners_.size()) ||
      !within(slice.output_begin, slice.output_count, buffer_owners_.size()) ||
      !within(slice.input_view_begin, slice.input_view_count,
              buffer_views_.size()) ||
      !within(slice.output_view_begin, slice.output_view_count,
              buffer_views_.size()) ||
      !within(slice.kernel_view_begin, slice.kernel_view_count,
              kernel_views_.size()) ||
      !within(slice.input_transfer_begin, slice.input_transfer_count,
              view_transfers_.size()) ||
      !within(slice.output_transfer_begin, slice.output_transfer_count,
              view_transfers_.size())) {
    return false;
  }
  storage = CpuJobBindingStorage{
      .inputs = buffer_owners_.subspan(slice.input_begin, slice.input_count),
      .outputs = buffer_owners_.subspan(slice.output_begin, slice.output_count),
      .input_views =
          buffer_views_.subspan(slice.input_view_begin, slice.input_view_count),
      .output_views = buffer_views_.subspan(slice.output_view_begin,
                                            slice.output_view_count),
      .kernel_views = kernel_views_.subspan(slice.kernel_view_begin,
                                            slice.kernel_view_count),
      .input_transfers = view_transfers_.subspan(slice.input_transfer_begin,
                                                 slice.input_transfer_count),
      .output_transfers = view_transfers_.subspan(slice.output_transfer_begin,
                                                  slice.output_transfer_count),
  };
  return true;
}

bool CpuPreparedArena::view(const CpuWorkspaceSlice &slice,
                            CpuWorkspaceStorage &storage) noexcept {
  const auto within = [](const std::size_t begin, const std::size_t count,
                         const std::size_t capacity) noexcept {
    return begin <= capacity && count <= capacity - begin;
  };
  if (slice.workspace_count != 1u ||
      !within(slice.workspace_begin, slice.workspace_count,
              workspaces_.size()) ||
      !within(slice.buffer_begin, slice.buffer_count, buffer_owners_.size()) ||
      !within(slice.offset_begin, slice.offset_count,
              workspace_offsets_.size()) ||
      slice.buffer_count != slice.offset_count) {
    return false;
  }
  storage = CpuWorkspaceStorage{
      .workspace = workspaces_.data() + slice.workspace_begin,
      .buffers = buffer_owners_.subspan(slice.buffer_begin, slice.buffer_count),
      .offsets =
          workspace_offsets_.subspan(slice.offset_begin, slice.offset_count),
  };
  return true;
}

Result<std::shared_ptr<CpuPreparedArena>>
make_cpu_prepared_arena(const CpuPreparedArenaPlan &plan) {
  CpuPreparedArenaPlan expected{};
  expected.execution = plan.execution;
  expected.map_count = plan.map_count;
  expected.read_count = plan.read_count;
  expected.write_count = plan.write_count;
  expected.buffer_owner_count = plan.buffer_owner_count;
  expected.buffer_view_count = plan.buffer_view_count;
  expected.kernel_view_count = plan.kernel_view_count;
  expected.view_transfer_count = plan.view_transfer_count;
  expected.workspace_count = plan.workspace_count;
  expected.workspace_offset_count = plan.workspace_offset_count;
  if (!plan.layout.sealed ||
      !seal_cpu_prepared_arena_plan(expected, plan.layout.page_bytes) ||
      expected != plan) {
    return Result<std::shared_ptr<CpuPreparedArena>>::fail(
        Reason::CpuRuntimeInvalid);
  }
  try {
    auto arena = std::make_shared<CpuPreparedArena>();
    if (!arena->materialize(plan) || !arena->supports(plan.execution) ||
        arena->extent_bytes() != plan.layout.extent_bytes ||
        arena->committed_bytes() != plan.layout.committed_bytes) {
      return Result<std::shared_ptr<CpuPreparedArena>>::fail(
          Reason::BufferCapacity);
    }
    return Result<std::shared_ptr<CpuPreparedArena>>::success(std::move(arena));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<CpuPreparedArena>>::fail(
        Reason::BufferCapacity);
  }
}

} // namespace rund::compute::detail
