#pragma once

#include <kernel/program/skeleton/shape.hpp>
#include <kernel/program/skeleton/model.hpp>

#include <limits>

namespace rund::kernel::skeleton_detail {

template <std::size_t Rank>
[[nodiscard]] constexpr u64 UnitCount(const Space<Rank>& index_space) noexcept {
  u64 units = 0u;
  static_cast<void>(ShapeProduct(index_space.extent, units));
  return units;
}

template <std::size_t Rank>
[[nodiscard]] constexpr const char* ValidateSpaceReason(
    const Space<Rank>& index_space) noexcept {
  if (!index_space.valid) {
    return index_space.reason;
  }
  u64 units = 0u;
  if (!ShapeProduct(index_space.extent, units)) {
    return "skeleton_index_space_overflow";
  }
  return nullptr;
}

template <std::size_t Rank>
[[nodiscard]] constexpr const char* ValidatePartitionSpaceReason(
    const Space<Rank>& index_space) noexcept {
  if (const char* const reason = ValidateSpaceReason(index_space);
      reason != nullptr) {
    return reason;
  }
  if (UnitCount(index_space) > std::numeric_limits<u32>::max()) {
    return "skeleton_partition_space_exceeds_u32";
  }
  return nullptr;
}

[[nodiscard]] constexpr const char* ValidateAlignmentReason(
    const Alignment alignment) noexcept {
  if (!alignment.valid) {
    return alignment.reason;
  }
  if (alignment.units == 0u) {
    return "skeleton_alignment_zero";
  }
  return nullptr;
}

[[nodiscard]] constexpr const char* ValidatePartitionBoundaryAlignmentReason(
    const Partition& partition,
    const Alignment alignment) noexcept {
  if (const char* const reason = ValidateAlignmentReason(alignment);
      reason != nullptr) {
    return reason;
  }
  if ((partition.begin % alignment.units) != 0u ||
      (partition.end % alignment.units) != 0u) {
    return "skeleton_partition_boundary_misaligned";
  }
  return nullptr;
}

} // namespace rund::kernel::skeleton_detail
