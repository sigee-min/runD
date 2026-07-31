#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include <node/accel/pick.hpp>

#include <node/accel/buffer.hpp>

#include "../../collective/scan/run.hpp"
#include "match.hpp"

#include <cstdio>

namespace node_accel_contract {

bool BackendRunsInclusiveScan(const rund::AccelDevice &pick) {
  return InclusiveScanMatchesReference<rund::kernel::u32>(
             pick, rund::kernel::ComputeScalar::Lane32,
             rund::kernel::ScanElement::U32,
             std::array<rund::kernel::u32, 8u>{1u, 1u, 2u, 3u, 5u, 8u, 13u,
                                               21u}) &&
         InclusiveScanMatchesReference<rund::kernel::u64>(
             pick, rund::kernel::ComputeScalar::Lane64,
             rund::kernel::ScanElement::U64,
             std::array<rund::kernel::u64, 8u>{2u, 3u, 5u, 7u, 11u, 13u, 17u,
                                               19u});
}

bool RequiredMetalRunsInclusiveScan() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Metal));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Metal);
  }
  return pick.api == rund::AccelApi::Metal && BackendRunsInclusiveScan(pick);
}

bool RequiredVulkanRunsInclusiveScan() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Vulkan));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Vulkan);
  }
  if (pick.api != rund::AccelApi::Vulkan) {
    return false;
  }
  const std::array<rund::kernel::u32, 8u> input{1u, 1u, 2u,  3u,
                                                5u, 8u, 13u, 21u};
  if (!collective::ScanMatchesCpuReference<rund::kernel::u32>(
          pick, rund::kernel::ScanElement::U32,
          rund::kernel::ComputeScalar::Lane32, input)) {
    return false;
  }
  const rund::RuntimeStats exclusive_stats =
      rund::node::accel::ReadRuntimeStats(pick);
  if (!exclusive_stats.ok || exclusive_stats.pipeline_compile_count != 3u ||
      exclusive_stats.pipeline_cache_hit_count != 0u) {
    std::fprintf(
        stderr,
        "vulkan exclusive scan cold cache mismatch: ok=%d reason=%s "
        "compile=%llu hit=%llu\n",
        exclusive_stats.ok, exclusive_stats.reason,
        static_cast<unsigned long long>(exclusive_stats.pipeline_compile_count),
        static_cast<unsigned long long>(
            exclusive_stats.pipeline_cache_hit_count));
    return false;
  }
  if (!InclusiveScanMatchesReference<rund::kernel::u32>(
          pick, rund::kernel::ComputeScalar::Lane32,
          rund::kernel::ScanElement::U32, input)) {
    return false;
  }
  const rund::RuntimeStats inclusive_stats =
      rund::node::accel::ReadRuntimeStats(pick);
  if (!inclusive_stats.ok || inclusive_stats.pipeline_compile_count != 1u ||
      inclusive_stats.pipeline_cache_hit_count != 2u) {
    std::fprintf(
        stderr,
        "vulkan inclusive scan shared prefix/offset cache mismatch: "
        "ok=%d reason=%s compile=%llu hit=%llu\n",
        inclusive_stats.ok, inclusive_stats.reason,
        static_cast<unsigned long long>(inclusive_stats.pipeline_compile_count),
        static_cast<unsigned long long>(
            inclusive_stats.pipeline_cache_hit_count));
    return false;
  }
  return InclusiveScanMatchesReference<rund::kernel::u64>(
      pick, rund::kernel::ComputeScalar::Lane64, rund::kernel::ScanElement::U64,
      std::array<rund::kernel::u64, 8u>{2u, 3u, 5u, 7u, 11u, 13u, 17u, 19u});
}

} // namespace node_accel_contract
