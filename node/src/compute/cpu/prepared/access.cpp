#include "internal.hpp"

#include "../../job/state.hpp"

#include <cstddef>
#include <span>

namespace rund::compute::detail {

bool CpuPreparedArena::supports(
    const CpuExecutionStoragePlan &plan) const noexcept {
  const bool tiles_fit =
      !plan.tiles.ok ||
      (cpu_prepared_detail::valid_tile_plan(plan.tiles) &&
       cpu_prepared_detail::valid_tile_plan(tile_capacity_) &&
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

} // namespace rund::compute::detail
