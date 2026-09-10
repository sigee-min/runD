#include "local.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace rund::measure::compute::virtual_window {
namespace {

[[nodiscard]] constexpr std::uint64_t
divide_up(const std::uint64_t value, const std::uint64_t divisor) noexcept {
  return value / divisor + static_cast<std::uint64_t>(value % divisor != 0u);
}

[[nodiscard]] bool valid_output(const std::span<const std::int32_t> values,
                                const std::size_t active) noexcept {
  if (active == 0u || active > values.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < active; ++index) {
    if (values[index] != expected_value(index)) {
      return false;
    }
  }
  const std::span<const std::byte> bytes = std::as_bytes(values);
  return std::all_of(bytes.begin() + static_cast<std::ptrdiff_t>(
                                         active * sizeof(std::int32_t)),
                     bytes.end(), [](const std::byte value) {
                       return value == virtual_residency::TailPoison;
                     });
}

[[nodiscard]] std::uint64_t
content_hash(const std::span<const std::int32_t> values,
             const std::size_t active) noexcept {
  constexpr std::uint64_t offset = 1'469'598'103'934'665'603ull;
  constexpr std::uint64_t prime = 1'099'511'628'211ull;
  std::uint64_t hash = offset;
  for (const std::byte value : std::as_bytes(values.first(active))) {
    hash ^= std::to_integer<std::uint8_t>(value);
    hash *= prime;
  }
  return hash;
}

[[nodiscard]] double microseconds(const Clock::duration duration) noexcept {
  return std::chrono::duration<double, std::micro>(duration).count();
}

[[nodiscard]] bool
same_counter_shape(const ::rund::compute::MemoryCounter left,
                   const ::rund::compute::MemoryCounter right) noexcept {
  return left.current == right.current && left.peak == right.peak &&
         left.budget == right.budget;
}

[[nodiscard]] bool
same_fixed_memory(const ::rund::compute::MemoryStats &left,
                  const ::rund::compute::MemoryStats &right) noexcept {
  return left.backend == right.backend && left.scope == right.scope &&
         same_counter_shape(left.host, right.host) &&
         same_counter_shape(left.frame, right.frame) &&
         same_counter_shape(left.tile, right.tile) &&
         same_counter_shape(left.resident, right.resident) &&
         same_counter_shape(left.staging, right.staging) &&
         same_counter_shape(left.device, right.device) &&
         same_counter_shape(left.transfer, right.transfer);
}

[[nodiscard]] bool observe(Prepared &prepared,
                           const std::size_t active) noexcept {
  const auto read = prepared.output->read(
      0u, std::as_writable_bytes(std::span{prepared.observed}));
  return read && valid_output(prepared.observed, active);
}

[[nodiscard]] bool run_abba(Prepared &cpu, Prepared &metal,
                            const std::size_t active,
                            virtual_residency::WallSamples &cpu_samples,
                            virtual_residency::WallSamples &metal_samples) {
  for (std::size_t cycle = 0u; cycle < WarmSamples / 2u; ++cycle) {
    const auto run = [&](Prepared &prepared,
                         virtual_residency::WallSamples &samples,
                         const std::size_t ordinal) noexcept {
      const auto begin = Clock::now();
      const auto status = prepared.pipeline.run(active);
      const auto end = Clock::now();
      samples.microseconds[ordinal] = microseconds(end - begin);
      return static_cast<bool>(status);
    };
    const std::size_t first = cycle * 2u;
    const std::size_t second = first + 1u;
    if (!run(cpu, cpu_samples, first) || !run(metal, metal_samples, first) ||
        !run(metal, metal_samples, second) || !run(cpu, cpu_samples, second)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool
valid_profile(const Prepared &prepared, const Profile &profile,
              const std::size_t active, const std::uint64_t epochs,
              const ::rund::compute::MemoryStats &warm_memory) noexcept {
  const auto &stats = profile.execution();
  const auto &residency = stats.pipeline.residency;
  const std::uint64_t active_pages = divide_up(active, PageElements);
  const std::uint64_t backing_bytes = active * sizeof(std::int32_t);
  const bool cpu = prepared.backend == Backend::Cpu;
  const auto &transfers = stats.transfer_submissions;
  const bool physical_zero =
      stats.uploaded_bytes == 0u && stats.downloaded_bytes == 0u &&
      transfers.host_to_device == 0u && transfers.device_to_host == 0u &&
      transfers.device_to_device == 0u;
  const bool backend_exact =
      cpu ? stats.command_submits == 0u && stats.command_inflight_peak == 0u &&
                residency.window_handoff_count == 0u &&
                residency.window_batch_count == 0u &&
                residency.window_queue_call_count == 0u &&
                residency.overlap_ns == 0u && residency.h2d_overlap_ns == 0u &&
                residency.d2h_overlap_ns == 0u
          : stats.command_submits == epochs &&
                stats.command_inflight_peak >= 2u &&
                residency.window_handoff_count == 1u &&
                residency.window_batch_count == epochs &&
                residency.window_queue_call_count == epochs &&
                residency.stall_ns != 0u;
  return stats.backend == prepared.backend && profile.memory().available() &&
         profile.memory().backend == prepared.backend &&
         prepared.pipeline.plan() == prepared.plan &&
         prepared.plan.residency.logical_bytes ==
             LogicalCapacity * sizeof(std::int32_t) * 2u &&
         prepared.plan.residency.page_bytes == FrameBytes * 2u &&
         prepared.plan.residency.page_count == 12u &&
         prepared.plan.residency.frame_capacity == FrameCapacity &&
         prepared.plan.residency.epoch_count == 4u &&
         residency.logical_bytes == prepared.plan.residency.logical_bytes &&
         residency.active_count == active &&
         residency.page_bytes == prepared.plan.residency.page_bytes &&
         residency.page_count == prepared.plan.residency.page_count &&
         residency.frame_capacity == FrameCapacity &&
         residency.epoch_count == epochs &&
         residency.page_in_count + residency.cache_hit_count == active_pages &&
         residency.page_out_count == active_pages &&
         residency.page_out_bytes == backing_bytes &&
         residency.backing_write_bytes == backing_bytes &&
         residency.backing_read_bytes <= backing_bytes &&
         residency.plan_identity_hi == prepared.plan.residency.identity_hi &&
         residency.plan_identity_lo == prepared.plan.residency.identity_lo &&
         residency.failed_page ==
             ::rund::compute::ResidencyStats::no_failed_page &&
         residency.samples_allocation_free(WarmSamples) &&
         residency.directional_overlap_exact() &&
         stats.dispatches == active_pages && physical_zero && backend_exact &&
         stats.pipeline_compiles == 0u && stats.buffer_allocations == 0u &&
         stats.descriptor_pool_creations == 0u &&
         stats.descriptor_set_allocations == 0u &&
         stats.pipeline_cache_evictions == 0u &&
         same_fixed_memory(warm_memory, profile.memory());
}

struct Summary final {
  double p25{};
  double p50{};
  double p75{};
  double p95{};
  double mad{};
};

[[nodiscard]] Summary
summarize(virtual_residency::WallSamples &samples) noexcept {
  samples.sort();
  Summary result{
      .p25 = samples.microseconds[14u],
      .p50 = samples.p50(),
      .p75 = samples.microseconds[44u],
      .p95 = samples.p95(),
  };
  std::array<double, WarmSamples> deviations{};
  std::transform(samples.microseconds.begin(), samples.microseconds.end(),
                 deviations.begin(), [&](const double value) {
                   return value >= result.p50 ? value - result.p50
                                              : result.p50 - value;
                 });
  std::sort(deviations.begin(), deviations.end());
  result.mad =
      deviations[WarmSamples / 2u - 1u] +
      (deviations[WarmSamples / 2u] - deviations[WarmSamples / 2u - 1u]) / 2.0;
  return result;
}

void print_profile(const ::rund::compute::Stats &stats) {
  const auto &r = stats.pipeline.residency;
  const auto &t = stats.transfer_submissions;
  std::printf(
      ",%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
      "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
      "%u,%u",
      static_cast<unsigned long long>(r.epoch_count),
      static_cast<unsigned long long>(stats.dispatches),
      static_cast<unsigned long long>(stats.command_submits),
      static_cast<unsigned long long>(stats.command_inflight_peak),
      static_cast<unsigned long long>(r.window_handoff_count),
      static_cast<unsigned long long>(r.window_batch_count),
      static_cast<unsigned long long>(r.window_queue_call_count),
      static_cast<unsigned long long>(t.host_to_device),
      static_cast<unsigned long long>(t.device_to_host),
      static_cast<unsigned long long>(stats.uploaded_bytes),
      static_cast<unsigned long long>(stats.downloaded_bytes),
      static_cast<unsigned long long>(r.stall_ns),
      static_cast<unsigned long long>(r.overlap_ns),
      static_cast<unsigned long long>(r.h2d_overlap_ns),
      static_cast<unsigned long long>(r.d2h_overlap_ns),
      static_cast<unsigned long long>(r.backing_read_bytes),
      static_cast<unsigned long long>(r.backing_write_bytes),
      static_cast<unsigned long long>(r.page_in_count),
      static_cast<unsigned long long>(r.cache_hit_count),
      static_cast<unsigned long long>(r.page_out_count),
      static_cast<unsigned long long>(r.page_in_bytes),
      static_cast<unsigned long long>(r.page_out_bytes),
      static_cast<unsigned long long>(stats.submit_wait_ns),
      static_cast<unsigned long long>(stats.readback_ns),
      static_cast<unsigned long long>(stats.kernel_ns),
      static_cast<unsigned long long>(stats.kernel_samples), r.sampled_runs,
      r.allocation_free_runs);
}

} // namespace

[[nodiscard]] bool measure_cell(Prepared &cpu, Prepared &metal,
                                const std::size_t active,
                                const std::uint64_t epochs) noexcept {
  cpu.output->reset(virtual_residency::TailPoison);
  metal.output->reset(virtual_residency::TailPoison);
  if (!cpu.output->invalidate() || !metal.output->invalidate() ||
      !cpu.pipeline.run(active) || !metal.pipeline.run(active)) {
    return false;
  }
  const auto cpu_memory = cpu.pipeline.memory();
  const auto metal_memory = metal.pipeline.memory();
  if (!cpu.pipeline.begin_samples() || !metal.pipeline.begin_samples()) {
    return false;
  }
  virtual_residency::WallSamples cpu_samples{};
  virtual_residency::WallSamples metal_samples{};
  const bool ran = run_abba(cpu, metal, active, cpu_samples, metal_samples);
  const auto cpu_end = cpu.pipeline.end_samples();
  const auto metal_end = metal.pipeline.end_samples();
  if (!ran || !cpu_end || !metal_end || !observe(cpu, active) ||
      !observe(metal, active)) {
    return false;
  }
  auto cpu_profile = cpu.pipeline.profile();
  auto metal_profile = metal.pipeline.profile();
  const bool cpu_valid = cpu_profile && valid_profile(cpu, *cpu_profile, active,
                                                      epochs, cpu_memory);
  const bool metal_valid =
      metal_profile &&
      valid_profile(metal, *metal_profile, active, epochs, metal_memory);
  if (!cpu_valid || !metal_valid) {
    return false;
  }
  const std::uint64_t cpu_hash = content_hash(cpu.observed, active);
  const std::uint64_t metal_hash = content_hash(metal.observed, active);
  const auto &cpu_stats = cpu_profile->execution();
  const auto &metal_stats = metal_profile->execution();
  if (cpu_hash != metal_hash || cpu_stats.output_hash != cpu_hash ||
      metal_stats.output_hash != metal_hash ||
      cpu_stats.graph_hash != metal_stats.graph_hash ||
      cpu.plan.residency.logical_bytes != metal.plan.residency.logical_bytes ||
      cpu.plan.residency.page_bytes != metal.plan.residency.page_bytes ||
      cpu.plan.residency.page_count != metal.plan.residency.page_count ||
      cpu.plan.residency.frame_capacity !=
          metal.plan.residency.frame_capacity) {
    return false;
  }
  const Summary cpu_wall = summarize(cpu_samples);
  const Summary metal_wall = summarize(metal_samples);
  const double speedup =
      metal_wall.p50 == 0.0 ? 0.0 : cpu_wall.p50 / metal_wall.p50;
  const char *const classification =
      metal_wall.p75 < cpu_wall.p25   ? "metal_observed"
      : cpu_wall.p75 < metal_wall.p25 ? "cpu_observed"
                                      : "indeterminate";
  std::printf(
      "virtual_window,current_source_diagnostic,ok,%llu,%zu,%zu,%zu,%llu,"
      "%u,direct_pointwise_q2_4_coherent_required,"
      "public_window_receipt,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
      "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.9f,%s",
      static_cast<unsigned long long>(epochs), LogicalCapacity, active,
      PageElements,
      static_cast<unsigned long long>(divide_up(active, PageElements)),
      FrameCapacity, static_cast<unsigned long long>(cpu_stats.graph_hash),
      static_cast<unsigned long long>(cpu.plan.residency.identity_hi),
      static_cast<unsigned long long>(cpu.plan.residency.identity_lo),
      static_cast<unsigned long long>(metal.plan.residency.identity_hi),
      static_cast<unsigned long long>(metal.plan.residency.identity_lo),
      static_cast<unsigned long long>(cpu_hash),
      static_cast<unsigned long long>(metal_hash), cpu_wall.p25, cpu_wall.p50,
      cpu_wall.p75, cpu_wall.p95, cpu_wall.mad, metal_wall.p25, metal_wall.p50,
      metal_wall.p75, metal_wall.p95, metal_wall.mad, speedup, classification);
  print_profile(cpu_stats);
  print_profile(metal_stats);
  std::printf(",2,2\n");
  return true;
}

} // namespace rund::measure::compute::virtual_window
