#include "local.hpp"

#include "caps.hpp"

namespace rund::node::accel::detail {
namespace {

inline constexpr const char *kCpuSimdStrategyInfo[] = {
    "unknown", "scalar", "sse2", "avx2", "avx512", "neon",
};

[[nodiscard]] constexpr rund::kernel::CpuSimdStrategy
PickStrategy(const CpuProfile &profile) noexcept {
  if (profile.neon) {
    return rund::kernel::CpuSimdStrategy::Neon;
  }
  return rund::kernel::CpuSimdStrategy::Sse2;
}

} // namespace

[[nodiscard]] const char *
CpuSimdStrategyInfo(const rund::kernel::CpuSimdStrategy strategy) noexcept {
  const auto index = static_cast<unsigned>(strategy);
  return index < (sizeof(kCpuSimdStrategyInfo) /
                  sizeof(kCpuSimdStrategyInfo[0]))
             ? kCpuSimdStrategyInfo[index]
             : "unknown";
}

CpuProfile DetectCpu() noexcept {
  CpuProfile profile{};

#if (defined(__x86_64__) || defined(__i386__)) &&                              \
    (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  profile.sse2 = __builtin_cpu_supports("sse2") != 0;
  profile.ok = profile.sse2;
  profile.reason = profile.ok ? "ok" : profile.reason;
  profile.source = "builtin_cpu_supports";
#elif defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
  profile.neon = true;
  profile.ok = true;
  profile.reason = "ok";
  profile.source = "compile_target_neon";
#endif

  return profile;
}

rund::kernel::CpuCaps MakeCpuCaps(const CpuProfile &profile) noexcept {
  if (!profile.ok || (!profile.neon && !profile.sse2)) {
    return rund::kernel::CpuCaps{.reason = profile.reason};
  }

  const rund::kernel::CpuSimdStrategy strategy = PickStrategy(profile);
  const rund::kernel::u32 lane_bytes = rund::kernel::CpuSimdLaneBytes(strategy);
  const rund::kernel::u32 fixed_lane32_lanes =
      rund::kernel::CpuSimdFixedLane32Lanes(strategy);
  const rund::kernel::u32 fixed_lane64_lanes =
      rund::kernel::CpuSimdFixedLane64Lanes(strategy);
  if (lane_bytes != 16u || fixed_lane32_lanes == 0u ||
      fixed_lane64_lanes == 0u) {
    return rund::kernel::CpuCaps{.reason = "cpu_strategy_invalid"};
  }

  return rund::kernel::CpuCaps{
      .backend = rund::kernel::ComputeBackend::Cpu,
      .strategy = strategy,
      .lane_bytes = lane_bytes,
      .fixed_lane32_lanes = fixed_lane32_lanes,
      .fixed_lane64_lanes = fixed_lane64_lanes,
      .ok = true,
      .reason = "ok",
  };
}

rund::kernel::CpuCaps DetectCpuCaps() noexcept {
  return MakeCpuCaps(DetectCpu());
}

} // namespace rund::node::accel::detail
