#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

enum class ComputeBackend : u8 {
  Cpu = 1u,
  Metal = 2u,
  Vulkan = 3u,
};

enum class CpuSimdStrategy : u8 {
  Scalar = 1u,
  Sse2 = 2u,
  Avx2 = 3u,
  Avx512 = 4u,
  Neon = 5u,
};

struct CpuCaps {
  ComputeBackend backend = ComputeBackend::Cpu;
  CpuSimdStrategy strategy = CpuSimdStrategy::Scalar;
  u32 lane_bytes = 0u;
  u32 fixed_lane32_lanes = 0u;
  u32 fixed_lane64_lanes = 0u;
  bool ok = false;
  const char *reason = "cpu_caps_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

[[nodiscard]] constexpr bool
KnownCpuSimdStrategy(const CpuSimdStrategy strategy) noexcept {
  return strategy == CpuSimdStrategy::Scalar ||
         strategy == CpuSimdStrategy::Sse2 ||
         strategy == CpuSimdStrategy::Avx2 ||
         strategy == CpuSimdStrategy::Avx512 ||
         strategy == CpuSimdStrategy::Neon;
}

[[nodiscard]] constexpr u32
CpuSimdLaneBytes(const CpuSimdStrategy strategy) noexcept {
  switch (strategy) {
  case CpuSimdStrategy::Scalar:
    return 8u;
  case CpuSimdStrategy::Sse2:
  case CpuSimdStrategy::Neon:
    return 16u;
  case CpuSimdStrategy::Avx2:
    return 32u;
  case CpuSimdStrategy::Avx512:
    return 64u;
  }
  return 0u;
}

[[nodiscard]] constexpr u32
CpuSimdFixedLane32Lanes(const CpuSimdStrategy strategy) noexcept {
  return strategy == CpuSimdStrategy::Scalar ? 1u
                                             : CpuSimdLaneBytes(strategy) / 4u;
}

[[nodiscard]] constexpr u32
CpuSimdFixedLane64Lanes(const CpuSimdStrategy strategy) noexcept {
  return strategy == CpuSimdStrategy::Scalar ? 1u
                                             : CpuSimdLaneBytes(strategy) / 8u;
}

[[nodiscard]] constexpr bool CpuCapsValid(const CpuCaps &caps) noexcept {
  return caps.ok && caps.backend == ComputeBackend::Cpu &&
         KnownCpuSimdStrategy(caps.strategy) &&
         caps.lane_bytes == CpuSimdLaneBytes(caps.strategy) &&
         caps.fixed_lane32_lanes == CpuSimdFixedLane32Lanes(caps.strategy) &&
         caps.fixed_lane64_lanes == CpuSimdFixedLane64Lanes(caps.strategy);
}

} // namespace rund::kernel
