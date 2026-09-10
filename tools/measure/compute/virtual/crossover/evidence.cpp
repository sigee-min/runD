#include "internal.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace rund::measure::compute::virtual_crossover::detail {
namespace {

[[nodiscard]] std::int32_t expected_value(const std::size_t index,
                                          const std::size_t count,
                                          const std::size_t radius) noexcept {
  std::int32_t total = 0;
  for (std::size_t sample = 0u; sample < radius * 2u + 1u; ++sample) {
    const std::size_t raw = index + sample;
    const std::size_t selected =
        raw < radius ? 0u : std::min(raw - radius, count - 1u);
    total += seed_value(selected);
  }
  return total;
}

} // namespace

bool valid_output(const std::span<const std::int32_t> values,
                  const std::size_t active, const std::size_t radius) noexcept {
  if (active == 0u || active > values.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < active; ++index) {
    if (values[index] != expected_value(index, active, radius)) {
      return false;
    }
  }
  const auto bytes = std::as_bytes(values);
  const std::size_t active_bytes = active * sizeof(std::int32_t);
  return std::all_of(bytes.begin() + static_cast<std::ptrdiff_t>(active_bytes),
                     bytes.end(), [](const std::byte value) {
                       return value == virtual_residency::TailPoison;
                     });
}

std::uint64_t content_hash(const std::span<const std::int32_t> values,
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

double microseconds(const Clock::duration duration) noexcept {
  return std::chrono::duration<double, std::micro>(duration).count();
}

bool observe_output(PreparedPoint &point, const std::size_t active) noexcept {
  const auto read = point.output_backing->read(
      0u, std::as_writable_bytes(std::span{point.observed}));
  return read && valid_output(point.observed, active, point.radius);
}

bool cold_run(PreparedPoint &point, const std::size_t active,
              double &wall_us) noexcept {
  point.output_backing->reset(virtual_residency::TailPoison);
  if (!point.output_backing->invalidate()) {
    return false;
  }
  const auto begin = Clock::now();
  const auto status = point.pipeline.run(active);
  const auto end = Clock::now();
  wall_us = microseconds(end - begin);
  if (!status || !observe_output(point, active)) {
    return false;
  }
  const auto profile = point.pipeline.profile();
  return profile && profile->execution().output_hash ==
                        content_hash(point.observed, active);
}

bool valid_preparation(const Profile &profile, const Backend backend) noexcept {
  const auto &stats = profile.execution();
  const bool memory = profile.memory().available() &&
                      profile.memory().backend == backend &&
                      stats.backend == backend;
  if (!memory) {
    return false;
  }
  if (backend == Backend::Cpu) {
    return stats.pipeline.preparation_evidence ==
               ::rund::compute::PreparationEvidenceSource::NoNativeProducer &&
           stats.pipeline_compiles == 0u && stats.pipeline_cache_hits == 0u &&
           stats.pipeline_cache_evictions == 0u;
  }
  return backend == Backend::Metal &&
         stats.pipeline.preparation_evidence ==
             ::rund::compute::PreparationEvidenceSource::OwnerLocal &&
         stats.pipeline_compiles + stats.pipeline_cache_hits == 2u &&
         stats.pipeline_cache_evictions == 0u;
}

bool valid_warm(const PreparedPoint &point, const Profile &profile,
                const std::size_t active) noexcept {
  const auto &stats = profile.execution();
  const auto &residency = stats.pipeline.residency;
  constexpr std::uint64_t payload = CorePageElements;
  const std::uint64_t active_pages = divide_up(active, payload);
  const std::uint64_t page_count =
      divide_up(point.logical_count, static_cast<std::size_t>(payload));
  const std::uint64_t frame_capacity =
      std::min<std::uint64_t>(RequestedDeviceFrames, page_count);
  const std::uint64_t active_epochs = divide_up(active_pages, frame_capacity);
  // This frozen crossover uses the default Host-memory backing.  Recurrent
  // Window is intentionally reserved for serialized Persistent backing after
  // the same-workload packet proved its per-epoch terminal chain slower on
  // Host memory.
  const bool recurrent_window = false;
  const std::uint64_t page_bytes = input_frame_bytes(point.radius) * 2u;
  const std::uint64_t persistent_misses =
      residency.late_page_count + residency.prefetch_count;
  const bool tier_exact =
      point.backend == Backend::Cpu
          ? persistent_misses == residency.page_in_count &&
                residency.page_in_bytes == residency.backing_read_bytes
          : persistent_misses <= residency.page_in_count &&
                residency.backing_read_bytes <= residency.page_in_bytes;
  const bool transfer_exact =
      point.backend == Backend::Cpu
          ? stats.uploaded_bytes == 0u && stats.downloaded_bytes == 0u &&
                residency.overlap_ns == 0u && residency.h2d_overlap_ns == 0u &&
                residency.d2h_overlap_ns == 0u
          : stats.uploaded_bytes ==
                    (recurrent_window ? 0u
                                      : residency.page_in_count *
                                            input_frame_bytes(point.radius)) &&
                stats.downloaded_bytes == 0u && residency.d2h_overlap_ns == 0u;
  return stats.backend == point.backend && profile.memory().available() &&
         profile.memory().backend == point.backend &&
         point.plan.residency.logical_bytes ==
             point.logical_count * sizeof(std::int32_t) * 2u &&
         point.plan.residency.page_bytes == page_bytes &&
         point.plan.residency.page_count == page_count &&
         point.plan.residency.frame_capacity == frame_capacity &&
         residency.logical_bytes == point.plan.residency.logical_bytes &&
         residency.active_count == active &&
         residency.page_bytes == page_bytes &&
         residency.page_count == page_count &&
         residency.frame_capacity == frame_capacity &&
         residency.epoch_count == active_epochs &&
         residency.page_in_count + residency.cache_hit_count == active_pages &&
         residency.page_out_count == active_pages &&
         residency.backing_write_bytes == active * sizeof(std::int32_t) &&
         residency.page_out_bytes == active * sizeof(std::int32_t) &&
         residency.plan_identity_hi == point.plan.residency.identity_hi &&
         residency.plan_identity_lo == point.plan.residency.identity_lo &&
         residency.failed_page ==
             ::rund::compute::ResidencyStats::no_failed_page &&
         residency.window_handoff_count == (recurrent_window ? 1u : 0u) &&
         residency.window_batch_count ==
             (recurrent_window ? active_epochs : 0u) &&
         residency.window_queue_call_count ==
             (recurrent_window ? active_epochs : 0u) &&
         residency.samples_allocation_free(virtual_residency::WarmSamples) &&
         stats.pipeline_compiles == 0u && stats.buffer_allocations == 0u &&
         stats.descriptor_pool_creations == 0u &&
         stats.descriptor_set_allocations == 0u &&
         stats.dispatches == active_pages &&
         stats.command_submits ==
             (point.backend == Backend::Cpu ? 0u : active_epochs) &&
         stats.command_inflight_peak ==
             (point.backend == Backend::Cpu
                  ? 0u
                  : std::min<std::uint64_t>(active_epochs, 2u)) &&
         residency.stall_ns != 0u && residency.directional_overlap_exact() &&
         tier_exact && transfer_exact;
}

void summarize(BackendEvidence &evidence) noexcept {
  evidence.samples.sort();
  evidence.p25_us = evidence.samples.microseconds[14u];
  evidence.p50_us = evidence.samples.p50();
  evidence.p75_us = evidence.samples.microseconds[44u];
  evidence.p95_us = evidence.samples.p95();
  std::array<double, virtual_residency::WarmSamples> deviations{};
  std::transform(evidence.samples.microseconds.begin(),
                 evidence.samples.microseconds.end(), deviations.begin(),
                 [&](const double value) {
                   return value >= evidence.p50_us ? value - evidence.p50_us
                                                   : evidence.p50_us - value;
                 });
  std::sort(deviations.begin(), deviations.end());
  evidence.mad_us = deviations[virtual_residency::WarmSamples / 2u - 1u] +
                    (deviations[virtual_residency::WarmSamples / 2u] -
                     deviations[virtual_residency::WarmSamples / 2u - 1u]) /
                        2.0;
}

} // namespace rund::measure::compute::virtual_crossover::detail
