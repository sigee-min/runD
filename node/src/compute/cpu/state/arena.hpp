#pragma once

#include "../../../accel/cpu/simd/dispatch.hpp"
#include "../../job/binding.hpp"
#include "../arena.hpp"
#include "binding.hpp"
#include "collective.hpp"
#include "map.hpp"
#include "route.hpp"

#include "../../status.hpp"

#include <kernel/program/compute/tile/run.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>

namespace rund::compute::detail {

struct CpuExecutionStoragePlan final {
  kernel::ComputeTileRunStoragePlan tiles{};
  std::size_t map_scratch_count{};
  std::size_t simd_count{};
  std::size_t collective_total_count{};
  std::size_t collective_prefix_count{};
  std::size_t primitive_u32_count{};
  std::size_t primitive_u64_count{};
  std::size_t primitive_i32_count{};
  std::size_t primitive_i64_count{};
  std::size_t scatter_slot_count{};
  std::size_t transform_i32_count{};
  std::size_t transform_i64_count{};
  // Scratch descriptors are persistent per primitive and therefore additive;
  // their payload buffers remain the component-wise maximum above. Storage is
  // max_align_t padded for placement while payload preserves exact accounting.
  std::size_t primitive_object_storage_bytes{};
  std::size_t primitive_object_payload_bytes{};

  [[nodiscard]] constexpr bool
  operator==(const CpuExecutionStoragePlan &other) const noexcept {
    const auto &left_memory = tiles.retained.memory;
    const auto &right_memory = other.tiles.retained.memory;
    return tiles.failure_slot_capacity == other.tiles.failure_slot_capacity &&
           tiles.worker_capacity == other.tiles.worker_capacity &&
           tiles.ok == other.tiles.ok &&
           tiles.retained.ok == other.tiles.retained.ok &&
           left_memory.state_bytes == right_memory.state_bytes &&
           left_memory.workspace_bytes == right_memory.workspace_bytes &&
           left_memory.failure_slot_bytes == right_memory.failure_slot_bytes &&
           left_memory.worker_tile_bytes ==
               right_memory.worker_tile_bytes &&
           left_memory.async_context_bytes ==
               right_memory.async_context_bytes &&
           left_memory.total_bytes == right_memory.total_bytes &&
           map_scratch_count == other.map_scratch_count &&
           simd_count == other.simd_count &&
           collective_total_count == other.collective_total_count &&
           collective_prefix_count == other.collective_prefix_count &&
           primitive_u32_count == other.primitive_u32_count &&
           primitive_u64_count == other.primitive_u64_count &&
           primitive_i32_count == other.primitive_i32_count &&
           primitive_i64_count == other.primitive_i64_count &&
           scatter_slot_count == other.scatter_slot_count &&
           transform_i32_count == other.transform_i32_count &&
           transform_i64_count == other.transform_i64_count &&
           primitive_object_storage_bytes ==
               other.primitive_object_storage_bytes &&
           primitive_object_payload_bytes ==
               other.primitive_object_payload_bytes;
  }
};

struct CpuPreparedArenaPlan final {
  CpuArenaLayout layout{};
  CpuExecutionStoragePlan execution{};
  CpuArenaSegment tile_state{};
  CpuArenaSegment failure_slots{};
  CpuArenaSegment worker_tiles{};
  CpuArenaSegment worker_stats_partitions{};
  CpuArenaSegment worker_stats_start_offset_ns{};
  CpuArenaSegment worker_stats_elapsed_ns{};
  CpuArenaSegment worker_stats_tail_wait_ns{};
  CpuArenaSegment map_scratch{};
  CpuArenaSegment simd{};
  CpuArenaSegment collective_totals{};
  CpuArenaSegment collective_prefixes{};
  CpuArenaSegment primitive_u32{};
  CpuArenaSegment primitive_u64{};
  CpuArenaSegment primitive_i32{};
  CpuArenaSegment primitive_i64{};
  CpuArenaSegment scatter_keys{};
  CpuArenaSegment scatter_marks{};
  CpuArenaSegment scatter_epoch{};
  CpuArenaSegment transform_i32{};
  CpuArenaSegment transform_i64{};
  CpuArenaSegment primitive_objects{};
  CpuArenaSegment maps{};
  CpuArenaSegment reads{};
  CpuArenaSegment writes{};
  CpuArenaSegment buffer_owners{};
  CpuArenaSegment buffer_views{};
  CpuArenaSegment kernel_views{};
  CpuArenaSegment view_transfers{};
  CpuArenaSegment workspaces{};
  CpuArenaSegment workspace_offsets{};
  std::size_t map_count{};
  std::size_t read_count{};
  std::size_t write_count{};
  std::size_t buffer_owner_count{};
  std::size_t buffer_view_count{};
  std::size_t kernel_view_count{};
  std::size_t view_transfer_count{};
  std::size_t workspace_count{};
  std::size_t workspace_offset_count{};

  [[nodiscard]] constexpr bool
  operator==(const CpuPreparedArenaPlan &) const noexcept = default;
};

// One non-movable mapping owns every mutable CPU preparation and execution
// slice. Program and Job state retain views into this owner; no sibling arena
// or per-step allocation is permitted.
class CpuPreparedArena final {
public:
  CpuPreparedArena() noexcept = default;
  CpuPreparedArena(const CpuPreparedArena &) = delete;
  CpuPreparedArena &operator=(const CpuPreparedArena &) = delete;
  CpuPreparedArena(CpuPreparedArena &&) = delete;
  CpuPreparedArena &operator=(CpuPreparedArena &&) = delete;
  ~CpuPreparedArena();

  [[nodiscard]] bool
  supports(const CpuExecutionStoragePlan &plan) const noexcept;
  [[nodiscard]] kernel::ComputeTileRunStorageView tile_storage() noexcept;
  [[nodiscard]] std::span<std::max_align_t> map_scratch() noexcept {
    return map_scratch_;
  }
  [[nodiscard]] std::span<CpuSimdCount> simd() noexcept { return simd_; }
  [[nodiscard]] std::span<CpuCollectiveWide> collective_totals() noexcept {
    return collective_totals_;
  }
  [[nodiscard]] std::span<CpuCollectiveWide> collective_prefixes() noexcept {
    return collective_prefixes_;
  }
  [[nodiscard]] std::span<kernel::u32> primitive_u32() noexcept {
    return primitive_u32_;
  }
  [[nodiscard]] std::span<kernel::u64> primitive_u64() noexcept {
    return primitive_u64_;
  }
  [[nodiscard]] std::span<kernel::i32> primitive_i32() noexcept {
    return primitive_i32_;
  }
  [[nodiscard]] std::span<kernel::i64> primitive_i64() noexcept {
    return primitive_i64_;
  }
  [[nodiscard]] std::span<kernel::u32> scatter_keys() noexcept {
    return scatter_keys_;
  }
  [[nodiscard]] std::span<kernel::u32> scatter_marks() noexcept {
    return scatter_marks_;
  }
  [[nodiscard]] kernel::u32 *scatter_epoch() noexcept {
    return scatter_epoch_.size() == 1u ? scatter_epoch_.data() : nullptr;
  }
  [[nodiscard]] std::span<kernel::i32>
  claim_transform_i32(std::size_t count) noexcept;
  [[nodiscard]] std::span<kernel::i64>
  claim_transform_i64(std::size_t count) noexcept;
  template <class Scratch>
  [[nodiscard]] Scratch *claim_primitive_object() noexcept {
    static_assert(std::is_nothrow_default_constructible_v<Scratch>);
    static_assert(std::is_trivially_destructible_v<Scratch>);
    static_assert(alignof(Scratch) <= alignof(std::max_align_t));
    constexpr std::size_t alignment = alignof(std::max_align_t);
    constexpr std::size_t storage_bytes =
        (sizeof(Scratch) + alignment - 1u) / alignment * alignment;
    if (primitive_objects_claimed_ > primitive_objects_.size() ||
        storage_bytes >
            primitive_objects_.size() - primitive_objects_claimed_) {
      return nullptr;
    }
    void *const address =
        primitive_objects_.data() + primitive_objects_claimed_;
    primitive_objects_claimed_ += storage_bytes;
    return std::construct_at(static_cast<Scratch *>(address));
  }
  [[nodiscard]] bool
  claims_complete(const CpuExecutionStoragePlan &plan) const noexcept {
    return transform_i32_claimed_ == plan.transform_i32_count &&
           transform_i64_claimed_ == plan.transform_i64_count &&
           primitive_objects_claimed_ == plan.primitive_object_storage_bytes;
  }
  [[nodiscard]] bool
  view(const CpuRunRouteSlice &slice, std::span<CpuMapRoute> &maps,
       std::span<node::accel::cpu_simd_detail::CpuSimdReadBinding> &reads,
       std::span<node::accel::cpu_simd_detail::CpuSimdWriteBinding>
           &writes) noexcept;
  [[nodiscard]] bool view(const CpuJobBindingSlice &slice,
                          CpuJobBindingStorage &storage) noexcept;
  [[nodiscard]] bool view(const CpuWorkspaceSlice &slice,
                          CpuWorkspaceStorage &storage) noexcept;
  [[nodiscard]] std::uint64_t extent_bytes() const noexcept {
    return mapping_.extent_bytes();
  }
  [[nodiscard]] std::uint64_t committed_bytes() const noexcept {
    return mapping_.committed_bytes();
  }
  [[nodiscard]] std::uint64_t payload_host_bytes() const noexcept {
    return payload_host_bytes_;
  }
  [[nodiscard]] std::uint64_t payload_tile_bytes() const noexcept {
    return payload_tile_bytes_;
  }

private:
  friend Result<std::shared_ptr<CpuPreparedArena>>
  make_cpu_prepared_arena(const CpuPreparedArenaPlan &plan);

  [[nodiscard]] bool materialize(const CpuPreparedArenaPlan &plan) noexcept;
  void clear() noexcept;
  CpuArenaMapping mapping_{};
  kernel::ComputeTileRunStoragePlan tile_capacity_{};
  std::span<kernel::ComputeTileRunStorage> tile_state_{};
  std::span<const char *> failure_slots_{};
  std::span<kernel::u32> worker_tiles_{};
  std::span<kernel::u32> worker_stats_partitions_{};
  std::span<kernel::u64> worker_stats_start_offset_ns_{};
  std::span<kernel::u64> worker_stats_elapsed_ns_{};
  std::span<kernel::u64> worker_stats_tail_wait_ns_{};
  std::span<std::max_align_t> map_scratch_{};
  std::span<CpuSimdCount> simd_{};
  std::span<CpuCollectiveWide> collective_totals_{};
  std::span<CpuCollectiveWide> collective_prefixes_{};
  std::span<kernel::u32> primitive_u32_{};
  std::span<kernel::u64> primitive_u64_{};
  std::span<kernel::i32> primitive_i32_{};
  std::span<kernel::i64> primitive_i64_{};
  std::span<kernel::u32> scatter_keys_{};
  std::span<kernel::u32> scatter_marks_{};
  std::span<kernel::u32> scatter_epoch_{};
  std::span<kernel::i32> transform_i32_{};
  std::span<kernel::i64> transform_i64_{};
  std::span<std::byte> primitive_objects_{};
  std::span<CpuMapRoute> maps_{};
  std::span<node::accel::cpu_simd_detail::CpuSimdReadBinding> reads_{};
  std::span<node::accel::cpu_simd_detail::CpuSimdWriteBinding> writes_{};
  std::span<std::shared_ptr<BufferState>> buffer_owners_{};
  std::span<JobBufferView> buffer_views_{};
  std::span<node::accel::detail::KernelViewSlot> kernel_views_{};
  std::span<CpuViewTransfer> view_transfers_{};
  std::span<JobWorkspace> workspaces_{};
  std::span<std::size_t> workspace_offsets_{};
  std::size_t transform_i32_claimed_{};
  std::size_t transform_i64_claimed_{};
  std::size_t primitive_objects_claimed_{};
  std::uint64_t payload_host_bytes_{};
  std::uint64_t payload_tile_bytes_{};
};

} // namespace rund::compute::detail
