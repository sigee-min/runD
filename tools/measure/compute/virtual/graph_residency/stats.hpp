#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace rund::measure::compute::virtual_graph_residency {

inline constexpr std::size_t SampleCount = 60u;

// Phase values are durations, never absolute clock marks.  They are
// available only when every warm invocation produced the ordered lifecycle.
struct PhaseSample final {
  std::array<double, SampleCount> samples{};
  double p25{};
  double p50{};
  double p75{};
  double p95{};
  double mad{};
  std::size_t valid{};
};

struct PhaseTiming final {
  PhaseSample pre{};
  PhaseSample queue{};
  PhaseSample finish{};
  PhaseSample post{};
};

struct Timing final {
  std::array<double, SampleCount> samples{};
  double p25{};
  double p50{};
  double p75{};
  double p95{};
  double mad{};
  PhaseTiming phases{};
};

[[nodiscard]] inline std::size_t
timing_rank(const std::size_t count, const std::size_t percentile) noexcept {
  return (count * percentile + 99u) / 100u - 1u;
}

[[nodiscard]] inline bool summarize_timing(Timing &timing) noexcept {
  for (const double sample : timing.samples) {
    if (!std::isfinite(sample) || sample <= 0.0) {
      return false;
    }
  }
  auto sorted = timing.samples;
  std::sort(sorted.begin(), sorted.end());
  const double p50 = sorted[timing_rank(SampleCount, 50u)];
  timing.p25 = sorted[timing_rank(SampleCount, 25u)];
  timing.p50 = p50;
  timing.p75 = sorted[timing_rank(SampleCount, 75u)];
  timing.p95 = sorted[timing_rank(SampleCount, 95u)];
  std::array<double, SampleCount> deviations{};
  for (std::size_t index = 0u; index < SampleCount; ++index) {
    deviations[index] = std::abs(timing.samples[index] - p50);
  }
  std::sort(deviations.begin(), deviations.end());
  timing.mad = deviations[timing_rank(SampleCount, 50u)];
  return std::isfinite(timing.p25) && std::isfinite(timing.p50) &&
         std::isfinite(timing.p75) && std::isfinite(timing.p95) &&
         std::isfinite(timing.mad);
}

[[nodiscard]] inline bool
timing_summary_matches(const Timing &timing) noexcept {
  Timing expected = timing;
  if (!summarize_timing(expected)) {
    return false;
  }
  return timing.p25 == expected.p25 && timing.p50 == expected.p50 &&
         timing.p75 == expected.p75 && timing.p95 == expected.p95 &&
         timing.mad == expected.mad;
}

[[nodiscard]] inline bool summarize_phase(PhaseSample &phase) noexcept {
  if (phase.valid != SampleCount) {
    phase.p25 = 0.0;
    phase.p50 = 0.0;
    phase.p75 = 0.0;
    phase.p95 = 0.0;
    phase.mad = 0.0;
    return false;
  }
  for (const double sample : phase.samples) {
    if (!std::isfinite(sample) || sample < 0.0) {
      phase.p25 = 0.0;
      phase.p50 = 0.0;
      phase.p75 = 0.0;
      phase.p95 = 0.0;
      phase.mad = 0.0;
      return false;
    }
  }
  auto sorted = phase.samples;
  std::sort(sorted.begin(), sorted.end());
  const double p50 = sorted[timing_rank(SampleCount, 50u)];
  phase.p25 = sorted[timing_rank(SampleCount, 25u)];
  phase.p50 = p50;
  phase.p75 = sorted[timing_rank(SampleCount, 75u)];
  phase.p95 = sorted[timing_rank(SampleCount, 95u)];
  std::array<double, SampleCount> deviations{};
  for (std::size_t index = 0u; index < SampleCount; ++index) {
    deviations[index] = std::abs(phase.samples[index] - p50);
  }
  std::sort(deviations.begin(), deviations.end());
  phase.mad = deviations[timing_rank(SampleCount, 50u)];
  return std::isfinite(phase.p25) && std::isfinite(phase.p50) &&
         std::isfinite(phase.p75) && std::isfinite(phase.p95) &&
         std::isfinite(phase.mad);
}

[[nodiscard]] inline bool summarize_phases(PhaseTiming &timing) noexcept {
  bool ok = summarize_phase(timing.pre);
  ok = summarize_phase(timing.queue) && ok;
  ok = summarize_phase(timing.finish) && ok;
  ok = summarize_phase(timing.post) && ok;
  return ok;
}

[[nodiscard]] inline bool
phase_summary_matches(const PhaseSample &phase) noexcept {
  if (phase.valid != SampleCount) {
    return false;
  }
  PhaseSample expected = phase;
  if (!summarize_phase(expected)) {
    return false;
  }
  return phase.p25 == expected.p25 && phase.p50 == expected.p50 &&
         phase.p75 == expected.p75 && phase.p95 == expected.p95 &&
         phase.mad == expected.mad;
}

[[nodiscard]] inline bool phase_timing_available(
    const PhaseTiming &timing) noexcept {
  return phase_summary_matches(timing.pre) &&
         phase_summary_matches(timing.queue) &&
         phase_summary_matches(timing.finish) &&
         phase_summary_matches(timing.post);
}

} // namespace rund::measure::compute::virtual_graph_residency
