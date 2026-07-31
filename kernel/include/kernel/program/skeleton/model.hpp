#pragma once

#include <kernel/core/model.hpp>

#include <array>
#include <cstddef>

namespace rund::kernel {

template <std::size_t Rank>
using Index = std::array<u64, Rank>;

template <std::size_t Rank>
  requires(Rank > 0u)
struct Space {
  Index<Rank> extent{};
  bool valid = true;
  const char* reason = "pass";
};

struct Alignment {
  u32 units = 1u;
  bool valid = true;
  const char* reason = "pass";
};

enum class ViewAccessPattern : u8 {
  Unsupported,
  Contiguous,
  StridedAffine,
  BroadcastZeroStride,
};

enum class SkeletonPlanKind : u8 {
  Invalid,
  Empty,
  ContiguousLinear,
  AffineIndex,
  SerialLeftFold,
};

struct SkeletonResult {
  bool ok = false;
  const char* reason = "not_run";
  u64 visited_count = 0u;
  SkeletonPlanKind plan_kind = SkeletonPlanKind::Invalid;
  bool deterministic_traversal = false;
  bool contiguous_linear_traversal = false;
  bool global_row_major_order = false;
  bool partition_boundary_checked = false;
  bool partition_boundary_aligned = false;
  u32 boundary_alignment_units = 1u;
  u64 physical_tile_units = 0u;
  bool tile_order_preserves_row_major = false;
  bool vectorizable_shape = false;
  bool simd_measured = false;
  const char* vector_reason = "not_evaluated";
};

template <std::size_t Rank>
  requires(Rank > 0u)
struct SkeletonPlan {
  SkeletonPlanKind kind = SkeletonPlanKind::Invalid;
  Space<Rank> space{};
  u64 begin = 0u;
  u64 end = 0u;
  u64 unit_count = 0u;
  bool partitioned = false;
  bool global_row_major_order = true;
  bool boundary_alignment_checked = false;
  bool boundary_aligned = false;
  u32 boundary_alignment_units = 1u;
  u64 physical_tile_units = 0u;
  bool tile_order_preserves_row_major = true;
  bool vectorizable_shape = false;
  bool simd_measured = false;
  const char* reason = "not_planned";
};

} // namespace rund::kernel
