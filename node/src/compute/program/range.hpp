#pragma once

#include "../../accel/range_aggregate/model/plan.hpp"

#include <rund/compute/program/range.hpp>

#include <cstdint>
#include <span>

namespace rund::compute::detail {

[[nodiscard]] constexpr RangeKind
public_range_kind(const node::accel::detail::RangePath value) noexcept {
  using node::accel::detail::RangePath;
  switch (value) {
  case RangePath::Direct:
    return RangeKind::Direct;
  case RangePath::SharedHalo:
    return RangeKind::Shared;
  case RangePath::PrefixDifference:
    return RangeKind::Prefix;
  case RangePath::TiledDifference:
    return RangeKind::Tiled;
  case RangePath::BlockPrefixSuffix:
    return RangeKind::Block;
  }
  return RangeKind::Direct;
}

inline void append_program_range(const node::accel::detail::RangePlan &plan,
                                 const std::uint32_t node,
                                 const std::span<RangeInfo> rows,
                                 RangeSnapshot &snapshot) noexcept {
  if (!plan.ok()) {
    return;
  }
  const auto &candidate = plan.candidate();
  if (snapshot.written < rows.size()) {
    const auto source = plan.source_identity();
    const auto execution = plan.execution_identity();
    rows[snapshot.written] = RangeInfo{
        .node = node,
        .kind = public_range_kind(candidate.disposition()),
        .width = candidate.width(),
        .stages = static_cast<std::uint32_t>(plan.stage_count()),
        .shared_capacity = candidate.radius_capacity(),
        .scratch_bytes = plan.cost().scratch_bytes,
        .source = graph::Fingerprint{.hi = source.hi, .lo = source.lo},
        .execution = graph::Fingerprint{.hi = execution.hi, .lo = execution.lo},
    };
    ++snapshot.written;
  }
  ++snapshot.total;
}

} // namespace rund::compute::detail
