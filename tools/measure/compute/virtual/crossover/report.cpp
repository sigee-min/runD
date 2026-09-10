#include "internal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace rund::measure::compute::virtual_crossover::detail {
namespace {

[[nodiscard]] std::uint64_t
host_supply_hits(const ::rund::compute::ResidencyStats &residency) noexcept {
  const std::uint64_t misses =
      residency.late_page_count + residency.prefetch_count;
  return misses <= residency.page_in_count ? residency.page_in_count - misses
                                           : 0u;
}

[[nodiscard]] double ratio(const std::uint64_t numerator,
                           const std::uint64_t denominator) noexcept {
  return denominator == 0u ? 0.0
                           : static_cast<double>(numerator) /
                                 static_cast<double>(denominator);
}

} // namespace

void print_point(const std::size_t logical_count, const std::size_t radius,
                 const ActiveRatio active_ratio, const std::size_t active,
                 const PreparedPoint &cpu, const PreparedPoint &metal,
                 const BackendEvidence &cpu_evidence,
                 const BackendEvidence &metal_evidence) {
  const auto &cpu_stats = cpu_evidence.profile->execution();
  const auto &metal_stats = metal_evidence.profile->execution();
  const auto &cpu_residency = cpu_stats.pipeline.residency;
  const auto &metal_residency = metal_stats.pipeline.residency;
  const std::uint64_t active_pages = divide_up(active, CorePageElements);
  const std::uint64_t page_count = divide_up(logical_count, CorePageElements);
  const std::uint64_t host_capacity =
      std::min<std::uint64_t>(page_count, RequestedHostFrames);
  const std::uint64_t cpu_host_hits = host_supply_hits(cpu_residency);
  const std::uint64_t metal_host_hits = host_supply_hits(metal_residency);
  const double speedup = metal_evidence.p50_us == 0.0
                             ? 0.0
                             : cpu_evidence.p50_us / metal_evidence.p50_us;
  const char *const classification =
      metal_evidence.p75_us < cpu_evidence.p25_us   ? "metal_observed"
      : cpu_evidence.p75_us < metal_evidence.p25_us ? "cpu_observed"
                                                    : "indeterminate";
  const std::uint64_t semantic_additions = active * radius * 2u;
  const double semantic_additions_per_logical_io_byte =
      static_cast<double>(radius) / sizeof(std::int32_t);
  std::printf(
      "virtual_crossover,current_source_diagnostic,ok,%zu,%zu,%zu,%zu,%zu,"
      "%zu,%.9f,%zu,%llu,%.9f,%llu,%.9f",
      logical_count, radius, radius * 2u + 1u, active_ratio.numerator,
      active_ratio.denominator, active, ratio(active, logical_count),
      frame_elements(radius), static_cast<unsigned long long>(active_pages),
      ratio(active_pages, cpu.plan.residency.frame_capacity),
      static_cast<unsigned long long>(semantic_additions),
      semantic_additions_per_logical_io_byte);
  std::printf(
      ",%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu",
      static_cast<unsigned long long>(cpu.plan.residency.frame_capacity),
      static_cast<unsigned long long>(host_capacity),
      static_cast<unsigned long long>(cpu_stats.graph_hash),
      static_cast<unsigned long long>(cpu.plan.residency.identity_hi),
      static_cast<unsigned long long>(cpu.plan.residency.identity_lo),
      static_cast<unsigned long long>(metal.plan.residency.identity_hi),
      static_cast<unsigned long long>(metal.plan.residency.identity_lo),
      static_cast<unsigned long long>(cpu_evidence.output_hash),
      static_cast<unsigned long long>(metal_evidence.output_hash));
  std::printf(",%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
              "%.3f,%.3f,%.3f,%.9f,%s",
              cpu_evidence.prepare_us, metal_evidence.prepare_us,
              cpu_evidence.cold_us, metal_evidence.cold_us, cpu_evidence.p25_us,
              cpu_evidence.p50_us, cpu_evidence.p75_us, cpu_evidence.p95_us,
              cpu_evidence.mad_us, metal_evidence.p25_us, metal_evidence.p50_us,
              metal_evidence.p75_us, metal_evidence.p95_us,
              metal_evidence.mad_us, speedup, classification);
  const auto print_residency =
      [](const ::rund::compute::ResidencyStats &residency,
         const std::uint64_t host_hits) {
        std::printf(
            ",%llu,%llu,%.9f,%llu,%.9f,%llu,%llu,%llu,%llu,%llu,%llu,"
            "%llu,%llu,%llu,%llu,%llu",
            static_cast<unsigned long long>(residency.page_in_count),
            static_cast<unsigned long long>(residency.cache_hit_count),
            ratio(residency.cache_hit_count,
                  residency.page_in_count + residency.cache_hit_count),
            static_cast<unsigned long long>(host_hits),
            ratio(host_hits, residency.page_in_count),
            static_cast<unsigned long long>(residency.eviction_count),
            static_cast<unsigned long long>(residency.late_page_count),
            static_cast<unsigned long long>(residency.prefetch_count),
            static_cast<unsigned long long>(residency.backing_read_bytes),
            static_cast<unsigned long long>(residency.stall_ns),
            static_cast<unsigned long long>(residency.overlap_ns),
            static_cast<unsigned long long>(residency.h2d_overlap_ns),
            static_cast<unsigned long long>(residency.d2h_overlap_ns),
            static_cast<unsigned long long>(residency.window_handoff_count),
            static_cast<unsigned long long>(residency.window_batch_count),
            static_cast<unsigned long long>(residency.window_queue_call_count));
      };
  print_residency(cpu_residency, cpu_host_hits);
  print_residency(metal_residency, metal_host_hits);
  std::printf(",%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
              "%llu,%u,%u\n",
              static_cast<unsigned long long>(cpu_stats.uploaded_bytes),
              static_cast<unsigned long long>(cpu_stats.downloaded_bytes),
              static_cast<unsigned long long>(metal_stats.uploaded_bytes),
              static_cast<unsigned long long>(metal_stats.downloaded_bytes),
              static_cast<unsigned long long>(cpu_stats.command_submits),
              static_cast<unsigned long long>(metal_stats.command_submits),
              static_cast<unsigned long long>(cpu_stats.submit_wait_ns),
              static_cast<unsigned long long>(metal_stats.submit_wait_ns),
              static_cast<unsigned long long>(cpu_stats.readback_ns),
              static_cast<unsigned long long>(metal_stats.readback_ns),
              static_cast<unsigned long long>(cpu_stats.kernel_ns),
              static_cast<unsigned long long>(metal_stats.kernel_ns),
              cpu_residency.sampled_runs, cpu_residency.allocation_free_runs);
}

} // namespace rund::measure::compute::virtual_crossover::detail
