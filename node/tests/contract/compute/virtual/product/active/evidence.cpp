#include "evidence/local.hpp"
#include "local.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace rund_node_test_virtual::product::active {

using evidence_detail::ExactActiveStats;
using evidence_detail::ReportActiveFirstFalse;
using evidence_detail::ReportActiveStatsFirstFalse;

[[nodiscard]] int CheckActiveEvidence(ActiveFixture &fixture,
                                      const std::size_t active_count,
                                      const BackingFacts &input_before,
                                      const BackingFacts &output_before,
                                      const std::uint64_t allocations) {
  using namespace rund::compute;
  auto &prepared = *fixture.prepared;
  auto &input_backing = *fixture.input_backing;
  auto &output_backing = *fixture.output_backing;
  std::array<std::byte, LogicalBytes> observed{};
  if (!output_backing.observe(observed)) {
    return 10;
  }
  const Stats stats = prepared.stats();
  const auto profile = prepared.profile();
  const std::uint64_t pages =
      active_count / PageElements +
      static_cast<std::uint64_t>(active_count % PageElements != 0u);
  const BackingFacts input_delta = Delta(input_backing.facts(), input_before);
  const BackingFacts output_delta =
      Delta(output_backing.facts(), output_before);
  const auto expected = std::span{fixture.golden}.first(active_count);
  const bool capacity_same = SameCapacity(fixture.memory, prepared.memory());
  const bool stats_exact =
      ExactActiveStats(stats, fixture.backend, active_count);
  const bool content_exact = PrefixAndTail(observed, active_count);
  // Native DeviceVsm, Vulkan, and Metal Persistent work may allocate opaque
  // backend command records observed by the process-global allocator. Keep
  // producer-owned allocation checks strict for every other route.
  const RouteKind sample =
      ClassifyMode(fixture.backend, stats.pipeline.residency, pages);
  const bool opaque_native =
      sample == RouteKind::DeviceVsm || fixture.backend == Backend::Vulkan ||
      (fixture.backend == Backend::Metal && sample == RouteKind::Persistent);
  const bool owned_allocation_free = opaque_native || allocations == 0u;
  if (!profile || !owned_allocation_free || prepared.plan() != fixture.plan ||
      input_backing.identity() != fixture.input_identity ||
      output_backing.identity() != fixture.output_identity || !capacity_same ||
      !stats_exact ||
      profile->execution().pipeline.residency.active_count != active_count ||
      profile->execution().output_hash != stats.output_hash ||
      stats.output_hash != HashValues(expected) ||
      input_delta.read_count > (WarmRuns + 1u) * pages ||
      input_delta.read_bytes >
          (WarmRuns + 1u) * active_count * sizeof(std::int32_t) ||
      input_delta.write_count != 0u || input_delta.observation_count != 0u ||
      output_delta.write_count != (WarmRuns + 1u) * pages ||
      output_delta.write_bytes !=
          (WarmRuns + 1u) * active_count * sizeof(std::int32_t) ||
      output_delta.observation_count != 1u ||
      output_delta.observation_bytes != LogicalBytes ||
      input_delta.read_failure_count != 0u ||
      output_delta.write_failure_count != 0u ||
      output_delta.partial_write_bytes != 0u || !content_exact ||
      !input_backing.tail_poisoned() || !output_backing.tail_poisoned()) {
    if (!profile) {
      ReportActiveFirstFalse("profile.present", 0u, 1u);
    } else if (!owned_allocation_free) {
      ReportActiveFirstFalse("allocation_free", allocations, 0u);
    } else if (prepared.plan() != fixture.plan) {
      ReportActiveFirstFalse("plan.identity", 0u, 1u);
    } else if (input_backing.identity() != fixture.input_identity) {
      ReportActiveFirstFalse("backing.input_identity", 0u, 1u);
    } else if (output_backing.identity() != fixture.output_identity) {
      ReportActiveFirstFalse("backing.output_identity", 0u, 1u);
    } else if (!capacity_same) {
      ReportActiveFirstFalse("memory.capacity", 0u, 1u);
    } else if (!stats_exact) {
      ReportActiveStatsFirstFalse(stats, fixture.backend, active_count);
    } else if (profile->execution().pipeline.residency.active_count !=
               active_count) {
      ReportActiveFirstFalse(
          "profile.active_count",
          profile->execution().pipeline.residency.active_count, active_count);
    } else if (profile->execution().output_hash != stats.output_hash) {
      ReportActiveFirstFalse("profile.output_hash",
                             profile->execution().output_hash,
                             stats.output_hash);
    } else if (stats.output_hash != HashValues(expected)) {
      ReportActiveFirstFalse("stats.output_hash", stats.output_hash,
                             HashValues(expected));
    } else if (input_delta.read_count > (WarmRuns + 1u) * pages) {
      ReportActiveFirstFalse("backing.input_read_count", input_delta.read_count,
                             (WarmRuns + 1u) * pages);
    } else if (input_delta.read_bytes >
               (WarmRuns + 1u) * active_count * sizeof(std::int32_t)) {
      ReportActiveFirstFalse("backing.input_read_bytes", input_delta.read_bytes,
                             (WarmRuns + 1u) * active_count *
                                 sizeof(std::int32_t));
    } else if (input_delta.write_count != 0u) {
      ReportActiveFirstFalse("backing.input_write_count",
                             input_delta.write_count, 0u);
    } else if (input_delta.observation_count != 0u) {
      ReportActiveFirstFalse("backing.input_observation_count",
                             input_delta.observation_count, 0u);
    } else if (output_delta.write_count != (WarmRuns + 1u) * pages) {
      ReportActiveFirstFalse("backing.output_write_count",
                             output_delta.write_count, (WarmRuns + 1u) * pages);
    } else if (output_delta.write_bytes !=
               (WarmRuns + 1u) * active_count * sizeof(std::int32_t)) {
      ReportActiveFirstFalse(
          "backing.output_write_bytes", output_delta.write_bytes,
          (WarmRuns + 1u) * active_count * sizeof(std::int32_t));
    } else if (output_delta.observation_count != 1u) {
      ReportActiveFirstFalse("backing.output_observation_count",
                             output_delta.observation_count, 1u);
    } else if (output_delta.observation_bytes != LogicalBytes) {
      ReportActiveFirstFalse("backing.output_observation_bytes",
                             output_delta.observation_bytes, LogicalBytes);
    } else if (input_delta.read_failure_count != 0u) {
      ReportActiveFirstFalse("backing.input_read_failures",
                             input_delta.read_failure_count, 0u);
    } else if (output_delta.write_failure_count != 0u) {
      ReportActiveFirstFalse("backing.output_write_failures",
                             output_delta.write_failure_count, 0u);
    } else if (output_delta.partial_write_bytes != 0u) {
      ReportActiveFirstFalse("backing.output_partial_write_bytes",
                             output_delta.partial_write_bytes, 0u);
    } else if (!content_exact) {
      ReportActiveFirstFalse("content.exact", 0u, 1u);
    } else if (!input_backing.tail_poisoned()) {
      ReportActiveFirstFalse("backing.input_tail_poisoned", 0u, 1u);
    } else if (!output_backing.tail_poisoned()) {
      ReportActiveFirstFalse("backing.output_tail_poisoned", 0u, 1u);
    }
    std::fprintf(
        stderr,
        "virtual active backend=%u n=%zu alloc=%llu capacity=%u stats=%u "
        "content=%u hash=%llx expected=%llx read=%llu/%llu "
        "write=%llu/%llu observe=%llu/%llu sampled=%u/%u\n",
        static_cast<unsigned>(fixture.backend), active_count,
        static_cast<unsigned long long>(allocations),
        static_cast<unsigned>(capacity_same),
        static_cast<unsigned>(stats_exact),
        static_cast<unsigned>(content_exact),
        static_cast<unsigned long long>(stats.output_hash),
        static_cast<unsigned long long>(HashValues(expected)),
        static_cast<unsigned long long>(input_delta.read_count),
        static_cast<unsigned long long>(input_delta.read_bytes),
        static_cast<unsigned long long>(output_delta.write_count),
        static_cast<unsigned long long>(output_delta.write_bytes),
        static_cast<unsigned long long>(output_delta.observation_count),
        static_cast<unsigned long long>(output_delta.observation_bytes),
        stats.pipeline.residency.sampled_runs,
        stats.pipeline.residency.allocation_free_runs);
    std::fprintf(
        stderr,
        "virtual cache pages=%llu epochs=%llu load=%llu hit=%llu evict=%llu "
        "late=%llu pin=%llu pout=%llu stall=%llu overlap=%llu upload=%llu "
        "download=%llu submits=%llu writeback=%llu\n",
        static_cast<unsigned long long>(pages),
        static_cast<unsigned long long>(stats.pipeline.residency.epoch_count),
        static_cast<unsigned long long>(stats.pipeline.residency.page_in_count),
        static_cast<unsigned long long>(
            stats.pipeline.residency.cache_hit_count),
        static_cast<unsigned long long>(
            stats.pipeline.residency.eviction_count),
        static_cast<unsigned long long>(
            stats.pipeline.residency.late_page_count),
        static_cast<unsigned long long>(stats.pipeline.residency.page_in_bytes),
        static_cast<unsigned long long>(
            stats.pipeline.residency.page_out_bytes),
        static_cast<unsigned long long>(stats.pipeline.residency.stall_ns),
        static_cast<unsigned long long>(stats.pipeline.residency.overlap_ns),
        static_cast<unsigned long long>(stats.uploaded_bytes),
        static_cast<unsigned long long>(stats.downloaded_bytes),
        static_cast<unsigned long long>(stats.command_submits),
        static_cast<unsigned long long>(
            stats.pipeline.residency.page_out_count));
    std::fprintf(
        stderr,
        "virtual stats dispatch=%llu logical=%llu active=%llu page=%llu/%llu "
        "frames=%llu failed=%llu backing=%llu/%llu transfer=%llu/%llu/%llu\n",
        static_cast<unsigned long long>(stats.dispatches),
        static_cast<unsigned long long>(stats.pipeline.residency.logical_bytes),
        static_cast<unsigned long long>(stats.pipeline.residency.active_count),
        static_cast<unsigned long long>(stats.pipeline.residency.page_bytes),
        static_cast<unsigned long long>(stats.pipeline.residency.page_count),
        static_cast<unsigned long long>(
            stats.pipeline.residency.resident_frames_peak),
        static_cast<unsigned long long>(stats.pipeline.residency.failed_page),
        static_cast<unsigned long long>(
            stats.pipeline.residency.backing_read_bytes),
        static_cast<unsigned long long>(
            stats.pipeline.residency.backing_write_bytes),
        static_cast<unsigned long long>(
            stats.transfer_submissions.host_to_device),
        static_cast<unsigned long long>(
            stats.transfer_submissions.device_to_host),
        static_cast<unsigned long long>(
            stats.transfer_submissions.device_to_device));
    for (std::size_t index = 0u;
         index < std::min<std::size_t>(active_count, 40u); ++index) {
      std::int32_t value = 0;
      std::memcpy(&value, observed.data() + index * sizeof(value),
                  sizeof(value));
      if (value != fixture.golden[index]) {
        std::fprintf(stderr,
                     "virtual mismatch index=%zu value=%d expected=%d\n", index,
                     value, fixture.golden[index]);
        break;
      }
    }
    return 11;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::active
