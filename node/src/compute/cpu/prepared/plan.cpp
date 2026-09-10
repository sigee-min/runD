#include "internal.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>

namespace rund::compute::detail {

namespace cpu_prepared_detail {
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

} // namespace

bool valid_tile_plan(const kernel::ComputeTileRunStoragePlan &plan) noexcept {
  if (!plan.ok || !plan.retained.ok) {
    return false;
  }
  return same_tile_plan(
      plan, kernel::PlanComputeTileRunStorage(plan.failure_slot_capacity,
                                              plan.worker_capacity));
}

bool has_execution(const CpuExecutionStoragePlan &plan) noexcept {
  return plan.tiles.ok || plan.map_scratch_count != 0u ||
         plan.simd_count != 0u || plan.collective_total_count != 0u ||
         plan.collective_prefix_count != 0u || plan.primitive_u32_count != 0u ||
         plan.primitive_u64_count != 0u || plan.primitive_i32_count != 0u ||
         plan.primitive_i64_count != 0u || plan.scatter_slot_count != 0u ||
         plan.transform_i32_count != 0u || plan.transform_i64_count != 0u ||
         plan.primitive_object_storage_bytes != 0u;
}

std::uint64_t saturating_extent(const std::size_t count,
                                const std::size_t width) noexcept {
  std::uint64_t result = 0u;
  return kernel::checked::mul(static_cast<std::uint64_t>(count),
                              static_cast<std::uint64_t>(width), result)
             ? result
             : std::numeric_limits<std::uint64_t>::max();
}

std::uint64_t saturating_add(const std::uint64_t left,
                             const std::uint64_t right) noexcept {
  std::uint64_t result = 0u;
  return kernel::checked::add(left, right, result)
             ? result
             : std::numeric_limits<std::uint64_t>::max();
}

} // namespace cpu_prepared_detail

bool cpu_execution_storage_required(
    const CpuExecutionStoragePlan &plan) noexcept {
  return cpu_prepared_detail::has_execution(plan);
}

bool merge_cpu_execution_storage_plan(
    CpuExecutionStoragePlan &target,
    const CpuExecutionStoragePlan &source) noexcept {
  if ((source.tiles.ok &&
       !cpu_prepared_detail::valid_tile_plan(source.tiles)) ||
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
    if (!cpu_prepared_detail::valid_tile_plan(next.tiles)) {
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

} // namespace rund::compute::detail
