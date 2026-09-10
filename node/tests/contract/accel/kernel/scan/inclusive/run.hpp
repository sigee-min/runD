#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include <node/accel/pick.hpp>

#include <node/accel/buffer.hpp>

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "src/accel/scan/prefix.hpp"
#include "src/accel/vulkan/adapter/state.hpp"
#endif

#include "../../collective/scan/run.hpp"
#include "match.hpp"

#include <cstdio>
#include <limits>
#include <vector>

namespace node_accel_contract {

template <std::size_t N>
[[nodiscard]] bool InclusiveScanCarries(const rund::AccelDevice &pick) {
  std::array<rund::kernel::u64, N> input{};
  for (std::size_t i = 0u; i < N; ++i) {
    input[i] = (static_cast<rund::kernel::u64>(i % 5u) << 32u) | 0xffffffffull;
  }
  for (const rund::kernel::u64 block : {31u, 127u, 128u, 129u, 256u}) {
    if (!InclusiveScanMatchesReference(
            pick, rund::kernel::ComputeScalar::Lane64,
            rund::kernel::ScanElement::U64, input, nullptr, block)) {
      return false;
    }
  }
  return true;
}

inline bool BackendRunsInclusiveScan(const rund::AccelDevice &pick) {
  return InclusiveScanCarries<255u>(pick) &&
         InclusiveScanCarries<256u>(pick) &&
         InclusiveScanCarries<257u>(pick) &&
         InclusiveScanCarries<1025u>(pick) &&
         InclusiveScanMatchesReference<rund::kernel::u32>(
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

inline bool RequiredMetalRunsInclusiveScan() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Metal));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Metal);
  }
  if (pick.api != rund::AccelApi::Metal ||
      !InclusiveScanMatchesReference<rund::kernel::u32>(
          pick, rund::kernel::ComputeScalar::Lane32,
          rund::kernel::ScanElement::U32,
          std::array<rund::kernel::u32, 4u>{1u, 1u, 2u, 3u})) {
    return false;
  }
  const rund::RuntimeStats single = rund::node::accel::ReadRuntimeStats(pick);
  if (!single.outcome.ok ||
      single.run.allocations.pipeline_compile_count != 1u ||
      single.run.allocations.pipeline_cache_hit_count != 0u ||
      single.run.work.dispatch_count != 1u) {
    std::fprintf(
        stderr,
        "metal single-block scan cache mismatch: ok=%d reason=%s "
        "compile=%llu hit=%llu dispatch=%llu\n",
        single.outcome.ok, single.outcome.reason,
        static_cast<unsigned long long>(
            single.run.allocations.pipeline_compile_count),
        static_cast<unsigned long long>(
            single.run.allocations.pipeline_cache_hit_count),
        static_cast<unsigned long long>(single.run.work.dispatch_count));
    return false;
  }
  if (!InclusiveScanMatchesReference<rund::kernel::u32>(
          pick, rund::kernel::ComputeScalar::Lane32,
          rund::kernel::ScanElement::U32,
          std::array<rund::kernel::u32, 8u>{1u, 1u, 2u, 3u, 5u, 8u, 13u,
                                            21u})) {
    return false;
  }
  const rund::RuntimeStats multi = rund::node::accel::ReadRuntimeStats(pick);
  if (!multi.outcome.ok || multi.run.allocations.pipeline_compile_count != 2u ||
      multi.run.allocations.pipeline_cache_hit_count != 1u ||
      multi.run.work.dispatch_count != 3u) {
    std::fprintf(
        stderr,
        "metal multi-block scan cache mismatch: ok=%d reason=%s "
        "compile=%llu hit=%llu dispatch=%llu\n",
        multi.outcome.ok, multi.outcome.reason,
        static_cast<unsigned long long>(
            multi.run.allocations.pipeline_compile_count),
        static_cast<unsigned long long>(
            multi.run.allocations.pipeline_cache_hit_count),
        static_cast<unsigned long long>(multi.run.work.dispatch_count));
    return false;
  }
  return InclusiveScanMatchesReference<rund::kernel::u64>(
      pick, rund::kernel::ComputeScalar::Lane64, rund::kernel::ScanElement::U64,
      std::array<rund::kernel::u64, 8u>{2u, 3u, 5u, 7u, 11u, 13u, 17u, 19u});
}

inline bool RequiredVulkanRunsInclusiveScan() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Vulkan));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Vulkan);
  }
  if (pick.api != rund::AccelApi::Vulkan) {
    return false;
  }
  const std::array<rund::kernel::u32, 4u> single_input{1u, 1u, 2u, 3u};
  InclusiveScanRunCounters single_counters{};
  if (!InclusiveScanMatchesReference<rund::kernel::u32>(
          pick, rund::kernel::ComputeScalar::Lane32,
          rund::kernel::ScanElement::U32, single_input, &single_counters)) {
    return false;
  }
  const rund::RuntimeStats single_stats =
      rund::node::accel::ReadRuntimeStats(pick);
  if (!single_stats.outcome.ok ||
      single_stats.run.allocations.pipeline_compile_count != 1u ||
      single_stats.run.allocations.pipeline_cache_hit_count != 0u ||
      single_stats.run.work.dispatch_count != 1u ||
      single_counters.dispatch_count != 1u) {
    std::fprintf(
        stderr,
        "vulkan one-stage scan mismatch: ok=%d reason=%s compile=%llu "
        "hit=%llu dispatch=%llu evidence=%llu\n",
        single_stats.outcome.ok, single_stats.outcome.reason,
        static_cast<unsigned long long>(
            single_stats.run.allocations.pipeline_compile_count),
        static_cast<unsigned long long>(
            single_stats.run.allocations.pipeline_cache_hit_count),
        static_cast<unsigned long long>(single_stats.run.work.dispatch_count),
        static_cast<unsigned long long>(single_counters.dispatch_count));
    return false;
  }

  const std::array<rund::kernel::u32, 8u> input{1u, 1u, 2u,  3u,
                                                5u, 8u, 13u, 21u};
  InclusiveScanRunCounters multi_counters{};
  if (!InclusiveScanMatchesReference<rund::kernel::u32>(
          pick, rund::kernel::ComputeScalar::Lane32,
          rund::kernel::ScanElement::U32, input, &multi_counters)) {
    return false;
  }
  const rund::RuntimeStats multi_stats =
      rund::node::accel::ReadRuntimeStats(pick);
  if (!multi_stats.outcome.ok ||
      multi_stats.run.allocations.pipeline_compile_count != 2u ||
      multi_stats.run.allocations.pipeline_cache_hit_count != 1u ||
      multi_stats.run.work.dispatch_count != 3u ||
      multi_counters.dispatch_count != 3u) {
    std::fprintf(
        stderr,
        "vulkan three-stage scan mismatch: ok=%d reason=%s compile=%llu "
        "hit=%llu dispatch=%llu evidence=%llu\n",
        multi_stats.outcome.ok, multi_stats.outcome.reason,
        static_cast<unsigned long long>(
            multi_stats.run.allocations.pipeline_compile_count),
        static_cast<unsigned long long>(
            multi_stats.run.allocations.pipeline_cache_hit_count),
        static_cast<unsigned long long>(multi_stats.run.work.dispatch_count),
        static_cast<unsigned long long>(multi_counters.dispatch_count));
    return false;
  }

  if (!collective::ScanMatchesCpuReference<rund::kernel::u32>(
          pick, rund::kernel::ScanElement::U32,
          rund::kernel::ComputeScalar::Lane32, input)) {
    return false;
  }
  const rund::RuntimeStats exclusive_stats =
      rund::node::accel::ReadRuntimeStats(pick);
  if (!exclusive_stats.outcome.ok ||
      exclusive_stats.run.allocations.pipeline_compile_count != 1u ||
      exclusive_stats.run.allocations.pipeline_cache_hit_count != 2u ||
      exclusive_stats.run.work.dispatch_count != 3u) {
    std::fprintf(stderr,
                 "vulkan exclusive scan shared prefix/offset mismatch: ok=%d "
                 "reason=%s compile=%llu hit=%llu dispatch=%llu\n",
                 exclusive_stats.outcome.ok, exclusive_stats.outcome.reason,
                 static_cast<unsigned long long>(
                     exclusive_stats.run.allocations.pipeline_compile_count),
                 static_cast<unsigned long long>(
                     exclusive_stats.run.allocations.pipeline_cache_hit_count),
                 static_cast<unsigned long long>(
                     exclusive_stats.run.work.dispatch_count));
    return false;
  }

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using namespace rund::node::accel::detail;
  const auto *const adapter =
      static_cast<const VulkanAdapter *>(pick.backend.context);
  if (adapter == nullptr || adapter->device == VK_NULL_HANDLE ||
      adapter->physical_device == VK_NULL_HANDLE ||
      adapter->max_dispatch_groups == 0u) {
    std::fprintf(stderr, "vulkan scan chunk adapter unavailable\n");
    return false;
  }
  constexpr std::uint64_t MaximumChunkBoundaryElements = 1u << 20u;
  const std::uint64_t limit = adapter->max_dispatch_groups;
  const std::uint64_t chunk_count = limit + 1u;
  std::uint64_t payload_bytes = 0u;
  const bool storage_representable =
      limit != std::numeric_limits<std::uint32_t>::max() &&
      rund::kernel::checked::mul(chunk_count, sizeof(rund::kernel::u32),
                                 payload_bytes) &&
      payload_bytes <= adapter->storage_limit;
  if (storage_representable && chunk_count <= MaximumChunkBoundaryElements) {
    std::vector<rund::kernel::u32> chunk_input(
        static_cast<std::size_t>(chunk_count), 1u);
    scan_inclusive::Resources<rund::kernel::u32> resources =
        scan_inclusive::BuildResources(
            pick, rund::kernel::ComputeScalar::Lane32,
            rund::kernel::ScanElement::U32, chunk_input.data(),
            chunk_input.size(), 1u);
    const rund::kernel::ScanDesc desc{
        .op = rund::kernel::ScanOp::InclusiveSum,
        .element = rund::kernel::ScanElement::U32,
        .element_count = chunk_count,
        .block_size = 1u,
    };
    const RangePrefixExec prefix =
        PlanScanPrefixExecution(rund::kernel::PlanScan(desc));
    if (!resources.kernel.check.ok || !prefix.ok() ||
        prefix.stage_count() != 3u ||
        ScanPrefixDispatches(prefix, limit) != 5u) {
      std::fprintf(
          stderr,
          "vulkan scan chunk setup mismatch: kernel=%d reason=%s "
          "prefix=%d stages=%zu dispatches=%llu limit=%llu\n",
          resources.kernel.check.ok, resources.kernel.check.reason, prefix.ok(),
          prefix.stage_count(),
          static_cast<unsigned long long>(ScanPrefixDispatches(prefix, limit)),
          static_cast<unsigned long long>(limit));
      return false;
    }
    const auto bindings = scan_inclusive::Bindings(resources);
    const rund::AccelEvidence evidence = rund::node::accel::RunAccelKernel(
        resources.context, resources.kernel,
        rund::AccelRun{.bindings = bindings.data(),
                       .binding_count = bindings.size(),
                       .tile_count = chunk_count,
                       .fresh_evidence = true});
    const rund::RuntimeStats stats = rund::node::accel::ReadRuntimeStats(pick);
    std::vector<rund::kernel::u32> downloaded(
        static_cast<std::size_t>(chunk_count));
    const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
        resources.context, resources.write, downloaded.data(), payload_bytes);
    if (!evidence.outcome.ok || evidence.run.work.dispatch_count != 5u ||
        !stats.outcome.ok || stats.run.work.dispatch_count != 5u ||
        stats.run.allocations.pipeline_compile_count != 0u ||
        stats.run.allocations.pipeline_cache_hit_count != 3u || !download.ok) {
      std::fprintf(
          stderr,
          "vulkan scan chunk runtime mismatch: evidence=%d reason=%s "
          "dispatch=%llu stats=%d/%s/%llu compile=%llu hit=%llu "
          "download=%d/%s\n",
          evidence.outcome.ok, evidence.outcome.reason,
          static_cast<unsigned long long>(evidence.run.work.dispatch_count),
          stats.outcome.ok, stats.outcome.reason,
          static_cast<unsigned long long>(stats.run.work.dispatch_count),
          static_cast<unsigned long long>(
              stats.run.allocations.pipeline_compile_count),
          static_cast<unsigned long long>(
              stats.run.allocations.pipeline_cache_hit_count),
          download.ok, download.reason);
      return false;
    }
    for (std::size_t index = 0u; index < downloaded.size(); ++index) {
      if (downloaded[index] != static_cast<rund::kernel::u32>(index + 1u)) {
        std::fprintf(stderr,
                     "vulkan scan chunk output mismatch: index=%zu "
                     "actual=%u expected=%u\n",
                     index, downloaded[index],
                     static_cast<rund::kernel::u32>(index + 1u));
        return false;
      }
    }
  } else {
    std::fprintf(stderr,
                 "vulkan scan chunk boundary not executed: limit=%llu "
                 "elements=%llu bytes=%llu storage=%llu budget=%llu\n",
                 static_cast<unsigned long long>(limit),
                 static_cast<unsigned long long>(chunk_count),
                 static_cast<unsigned long long>(payload_bytes),
                 static_cast<unsigned long long>(adapter->storage_limit),
                 static_cast<unsigned long long>(MaximumChunkBoundaryElements));
  }
#endif

  return InclusiveScanMatchesReference<rund::kernel::u64>(
      pick, rund::kernel::ComputeScalar::Lane64, rund::kernel::ScanElement::U64,
      std::array<rund::kernel::u64, 8u>{2u, 3u, 5u, 7u, 11u, 13u, 17u, 19u});
}

} // namespace node_accel_contract
