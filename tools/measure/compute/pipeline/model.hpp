#pragma once

#include "../model.hpp"
#include "../pipeline.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>
#include <rund/compute/pipeline.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace rund::measure::compute {

using ::rund::compute::Stats;

struct Durations final {
  std::vector<double> wall;
  std::vector<double> submit;
  std::vector<double> kernel;
  std::vector<double> claim;
  std::vector<double> control;
  std::vector<double> read;
  std::vector<double> readback;

  void reserve(const std::size_t samples) {
    wall.reserve(samples);
    submit.reserve(samples);
    kernel.reserve(samples);
    claim.reserve(samples);
    control.reserve(samples);
    read.reserve(samples);
    readback.reserve(samples);
  }
};

struct ExecutionCounters final {
  std::uint64_t command_submits{};
  std::uint64_t dispatches{};
};

[[nodiscard]] inline const char *CommandPath(const Backend backend) noexcept {
  return backend == Backend::Metal ? "reusable_icb" : "immutable_primary";
}

inline void ObserveWarm(WarmCounters &total, const Stats &stats) noexcept {
  total.observe(stats);
}

inline void AccumulateWarm(WarmCounters &total, const WarmCounters &part) noexcept {
  total.observe(part);
}

[[nodiscard]] inline std::uint64_t
ContentHash(const std::span<const std::int32_t> values) noexcept {
  std::uint64_t hash = 14695981039346656037ull;
  for (const std::int32_t value : values) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    for (unsigned shift = 0u; shift != 32u; shift += 8u) {
      hash ^= (bits >> shift) & 0xffu;
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

[[nodiscard]] inline bool
ValidValues(const std::span<const std::int32_t> input,
            const std::span<const std::int32_t> output) noexcept {
  if (input.size() != output.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < input.size(); ++index) {
    if (output[index] != input[index] * 30 + 7) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool PipelineCounters(const Backend backend, const Stats &stats,
                                    const std::uint64_t fingerprint) noexcept {
  const auto &pipeline = stats.pipeline;
  const std::uint64_t control_commands = 1u;
  return stats.backend == backend && stats.command_submits == 1u &&
         stats.dispatches == 3u && stats.output_hash == 0u &&
         stats.graph_hash == fingerprint && pipeline.step_count == 3u &&
         pipeline.resource_count == 4u && pipeline.barrier_count == 2u &&
         pipeline.claim_conflict_count == 0u &&
         pipeline.verified_step_count == 3u &&
         pipeline.failed_step_index ==
             ::rund::compute::PipelineStats::no_failed_step &&
         pipeline.status_entry_count == 0u &&
         pipeline.control_byte_count == 80u &&
         pipeline.control_command_count == control_commands;
}

[[nodiscard]] inline bool
TimingUnavailable(const ::rund::compute::StepTiming &timing) noexcept {
  return !timing.available() && timing.duration_ns == 0u &&
         timing.sample_count == 0u &&
         timing.clock == ::rund::compute::StepClock::Unavailable &&
         timing.relation == ::rund::compute::StepTimingRelation::Unavailable;
}

[[nodiscard]] inline bool ProfileEvidenceValid(
    const Backend backend,
    const ::rund::compute::PipelineProfileSnapshot &snapshot,
    const std::span<const ::rund::compute::PipelineStepProfile> rows,
    const std::span<const ::rund::compute::graph::Fingerprint> programs,
    const std::uint64_t fingerprint,
    const std::uint64_t expected_work_items) noexcept {
  constexpr std::uint64_t declared_steps = 3u;
  constexpr std::uint64_t active_steps = 3u;
  constexpr std::uint64_t vulkan_map_width = 256u;
  constexpr std::uint64_t control_bytes =
      declared_steps * sizeof(::rund::compute::ControlStats);
  constexpr std::uint64_t timestamp_bytes =
      2u * active_steps * sizeof(std::uint64_t);
  const std::uint64_t expected_workgroups =
      expected_work_items / vulkan_map_width +
      (expected_work_items % vulkan_map_width == 0u ? 0u : 1u);
  if (rows.size() != declared_steps || programs.size() != declared_steps ||
      snapshot.written != declared_steps || snapshot.total != declared_steps ||
      snapshot.truncated() || !snapshot.observation.available() ||
      snapshot.observation.clock != ::rund::compute::StepClock::HostSteady ||
      snapshot.observation.relation !=
          ::rund::compute::StepTimingRelation::Exclusive ||
      snapshot.observation.sample_count != 1u ||
      !PipelineCounters(backend, snapshot.execution, fingerprint)) {
    return false;
  }
  bool unavailable = true;
  bool device = true;
  for (std::size_t index = 0u; index < rows.size(); ++index) {
    const ::rund::compute::PipelineStepProfile &row = rows[index];
    unavailable = unavailable && TimingUnavailable(row.timing);
    device =
        device && row.timing.available() && row.timing.sample_count == 1u &&
        row.timing.clock == ::rund::compute::StepClock::Device &&
        row.timing.relation == ::rund::compute::StepTimingRelation::NonAdditive;
    if (row.index != index || row.program != programs[index] ||
        !row.execution.available() || row.execution.sample_count != 1u ||
        row.execution.original_dispatches != 1u ||
        row.execution.final_dispatches != 1u ||
        row.execution.barrier_count != (index == 0u ? 0u : 1u) ||
        (backend == Backend::Vulkan &&
         (row.execution.workgroup_count != expected_workgroups ||
          row.execution.work_item_count != expected_work_items)) ||
        row.memory.backend != backend ||
        row.memory.scope != ::rund::compute::MemoryScope::Pipeline) {
      return false;
    }
  }
  if (backend == Backend::Metal) {
    return unavailable && snapshot.instrumentation_command_count == 0u &&
           snapshot.instrumentation_byte_count >= control_bytes;
  }
  const bool without_timestamps =
      unavailable && snapshot.instrumentation_command_count == 3u &&
      snapshot.instrumentation_byte_count == control_bytes;
  const bool with_timestamps =
      device &&
      snapshot.instrumentation_command_count == 4u + 2u * active_steps &&
      snapshot.instrumentation_byte_count == control_bytes + timestamp_bytes;
  return backend == Backend::Vulkan && (without_timestamps || with_timestamps);
}

[[nodiscard]] inline const char *ProfileClockName(
    const std::span<const ::rund::compute::PipelineStepProfile> rows) noexcept {
  const bool device =
      std::all_of(rows.begin(), rows.end(), [](const auto &row) {
        return row.timing.available() &&
               row.timing.clock == ::rund::compute::StepClock::Device;
      });
  return device ? "device" : "unavailable";
}

[[nodiscard]] inline bool
SameStatusIdentity(const ::rund::compute::Status left,
                   const ::rund::compute::Status right) noexcept {
  return left.reason() == right.reason() && left.code() == right.code() &&
         left.error() == right.error();
}

} // namespace rund::measure::compute
