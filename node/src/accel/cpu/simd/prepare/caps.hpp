#pragma once

#include <math32/simd/model.hpp>
#include <math64/simd/model.hpp>

namespace rund::node::accel::cpu_simd_detail {
namespace {

[[nodiscard]] bool SupportsPortableLaneShape(
    const rund::kernel::CpuCaps& caps,
    const ComputeScalar scalar) noexcept {
  if (caps.lane_bytes != 16u) { return false; }
  if (caps.strategy != rund::kernel::CpuSimdStrategy::Sse2 &&
      caps.strategy != rund::kernel::CpuSimdStrategy::Neon) {
    return false;
  }
  return (scalar == ComputeScalar::Lane32 &&
          caps.fixed_lane32_lanes == rund::math32::simd::LaneCount) ||
         (scalar == ComputeScalar::Lane64 &&
          caps.fixed_lane64_lanes == rund::math64::simd::LaneCount);
}

}  // namespace
}  // namespace rund::node::accel::cpu_simd_detail
