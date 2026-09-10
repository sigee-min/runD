#include "internal.hpp"

namespace rund::compute::detail {

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

} // namespace rund::compute::detail
