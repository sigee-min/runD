#pragma once

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>
#include <rund/compute/pipeline.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace package_device_program {

using namespace rund::compute;
inline constexpr std::size_t Capacity = 8u;
inline constexpr std::size_t CandidateCapacity = Capacity * 2u;
inline constexpr std::size_t IslandCount = 2u;
inline constexpr std::size_t EventCapacity = IslandCount;
inline std::uint64_t domain_authoring_calls{};
inline constexpr std::size_t MemoryEntryCapacity = 128u;

struct Evidence final {
  std::array<std::int32_t, Capacity> state{};
  std::array<std::int32_t, EventCapacity> events{};
  std::array<std::int32_t, EventCapacity> integrated{};
  std::array<std::int32_t, 2u> resolved{};
  std::uint32_t event_count{};
  std::int32_t state_hash{};
  std::uint64_t snapshot_hash{};
  std::uint64_t iterations{};
  std::uint64_t conflicts{};
  graph::Fingerprint pipeline_fingerprint{};
  Backend backend{Backend::Unavailable};
  Reason terminal_reason{Reason::Ok};
  Code terminal_code{Code::Ok};
  std::uint64_t generation{};
  std::uint64_t command_submits{};
  std::uint64_t dispatches{};
};

[[nodiscard]] constexpr bool same_counter(const MemoryCounter &left,
                                          const MemoryCounter &right) {
  return left.current == right.current && left.peak == right.peak &&
         left.cumulative == right.cumulative && left.reused == right.reused &&
         left.budget == right.budget;
}

[[nodiscard]] constexpr bool same_memory(const MemoryStats &left,
                                         const MemoryStats &right) {
  return left.backend == right.backend && left.scope == right.scope &&
         same_counter(left.host, right.host) &&
         same_counter(left.frame, right.frame) &&
         same_counter(left.tile, right.tile) &&
         same_counter(left.resident, right.resident) &&
         same_counter(left.staging, right.staging) &&
         same_counter(left.device, right.device) &&
         same_counter(left.transfer, right.transfer);
}

[[nodiscard]] inline bool same_memory_snapshot(
    const MemorySnapshot &left,
    const std::array<MemoryEntry, MemoryEntryCapacity> &left_entries,
    const MemorySnapshot &right,
    const std::array<MemoryEntry, MemoryEntryCapacity> &right_entries) {
  if (left.truncated() || right.truncated() || left.written != left.total ||
      right.written != right.total || left.written != right.written ||
      !same_memory(left.summary, right.summary)) {
    return false;
  }
  for (std::size_t index = 0u; index < left.written; ++index) {
    const MemoryEntry &a = left_entries[index];
    const MemoryEntry &b = right_entries[index];
    if (a.category != b.category || a.use != b.use || a.index != b.index ||
        !same_counter(a.bytes, b.bytes)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool same_evidence(const Evidence &left,
                                        const Evidence &right) noexcept {
  if (left.state != right.state || left.resolved != right.resolved ||
      left.event_count != right.event_count ||
      left.event_count > EventCapacity || left.state_hash != right.state_hash ||
      left.snapshot_hash != right.snapshot_hash ||
      left.iterations != right.iterations ||
      left.conflicts != right.conflicts) {
    return false;
  }
  const auto count = static_cast<std::size_t>(left.event_count);
  return std::equal(left.events.begin(), left.events.begin() + count,
                    right.events.begin()) &&
         std::equal(left.integrated.begin(), left.integrated.begin() + count,
                    right.integrated.begin());
}

[[nodiscard]] inline bool
same_execution_identity(const Evidence &left, const Evidence &right) noexcept {
  return same_evidence(left, right) &&
         left.pipeline_fingerprint == right.pipeline_fingerprint &&
         left.backend == right.backend &&
         left.terminal_reason == right.terminal_reason &&
         left.terminal_code == right.terminal_code &&
         left.generation == right.generation &&
         left.command_submits == right.command_submits &&
         left.dispatches == right.dispatches;
}

inline void add(std::uint64_t &target, const std::uint64_t value) noexcept {
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  target = value > maximum - target ? maximum : target + value;
}

inline void add(MemoryCounter &target, const MemoryCounter &value) noexcept {
  add(target.current, value.current);
  add(target.peak, value.peak);
  add(target.cumulative, value.cumulative);
  add(target.reused, value.reused);
  add(target.budget, value.budget);
}

[[nodiscard]] inline bool add(MemoryStats &target,
                              const MemoryStats &value) noexcept {
  if (target.backend != value.backend || target.scope != value.scope) {
    return false;
  }
  add(target.host, value.host);
  add(target.frame, value.frame);
  add(target.tile, value.tile);
  add(target.resident, value.resident);
  add(target.staging, value.staging);
  add(target.device, value.device);
  add(target.transfer, value.transfer);
  return true;
}

template <std::size_t N>
[[nodiscard]] bool
profile_memory_reconciles(const PipelineProfileSnapshot &profile,
                          const std::array<PipelineStepProfile, N> &rows,
                          const MemoryStats &owner) noexcept {
  MemoryStats total = profile.shared_memory;
  for (const PipelineStepProfile &row : rows) {
    if (!add(total, row.memory)) {
      return false;
    }
  }
  return same_memory(total, profile.memory) &&
         same_memory(profile.memory, owner);
}

[[nodiscard]] inline bool unavailable(const StepTiming &timing) noexcept {
  return !timing.available() && timing.duration_ns == 0u &&
         timing.sample_count == 0u && timing.clock == StepClock::Unavailable &&
         timing.relation == StepTimingRelation::Unavailable;
}

[[nodiscard]] inline bool measured(const StepTiming &timing,
                                   const StepClock clock,
                                   const StepTimingRelation relation) noexcept {
  return timing.available() && timing.sample_count == 1u &&
         timing.clock == clock && timing.relation == relation;
}

[[nodiscard]] inline bool valid_instrumentation(
    const Backend backend, const PipelineProfileSnapshot &profile,
    const std::span<const PipelineStepProfile> rows) noexcept {
  const std::uint64_t control_bytes =
      static_cast<std::uint64_t>(rows.size()) * 64u;
  if (backend == Backend::Cpu) {
    return profile.instrumentation_command_count == 0u &&
           profile.instrumentation_byte_count == 0u;
  }
  if (backend == Backend::Metal) {
    return profile.instrumentation_command_count == 0u &&
           profile.instrumentation_byte_count == control_bytes;
  }
  const bool device_timestamps =
      std::all_of(rows.begin(), rows.end(), [](const auto &row) {
        return measured(row.timing, StepClock::Device,
                        StepTimingRelation::NonAdditive);
      });
  const bool no_timestamps =
      std::all_of(rows.begin(), rows.end(),
                  [](const auto &row) { return unavailable(row.timing); });
  const std::uint64_t active = rows.size();
  return backend == Backend::Vulkan &&
         ((no_timestamps && profile.instrumentation_command_count == 3u &&
           profile.instrumentation_byte_count == control_bytes) ||
          (device_timestamps &&
           profile.instrumentation_command_count == 4u + 2u * active &&
           profile.instrumentation_byte_count == control_bytes + 16u * active));
}

template <std::size_t N>
[[nodiscard]] bool
valid_profile(const Backend backend, const PipelineProfileSnapshot &profile,
              const std::array<PipelineStepProfile, N> &rows,
              const std::array<graph::Fingerprint, N> &programs,
              const MemoryStats &owner,
              const std::uint64_t referenced_bytes) noexcept {
  if (profile.written != N || profile.total != N || profile.truncated() ||
      profile.execution.backend != backend ||
      profile.execution.pipeline.step_count != N ||
      profile.referenced_resource_bytes != referenced_bytes ||
      !profile.memory.available() || !profile.shared_memory.available() ||
      !profile.observation.available() ||
      profile.observation.sample_count != 1u ||
      profile.observation.clock != StepClock::HostSteady ||
      profile.observation.relation != StepTimingRelation::Exclusive ||
      !profile_memory_reconciles(profile, rows, owner) ||
      !valid_instrumentation(backend, profile, rows)) {
    return false;
  }
  std::uint64_t barriers{};
  for (std::size_t index = 0u; index < N; ++index) {
    const PipelineStepProfile &row = rows[index];
    if (row.index != index || row.program != programs[index] ||
        !row.execution.available() || row.execution.original_dispatches == 0u ||
        row.execution.final_dispatches == 0u) {
      return false;
    }
    add(barriers, row.execution.barrier_count);
    if (backend == Backend::Cpu) {
      if (!measured(row.timing, StepClock::HostSteady,
                    StepTimingRelation::Exclusive)) {
        return false;
      }
    } else if (backend == Backend::Metal) {
      if (!unavailable(row.timing)) {
        return false;
      }
    } else if (!(unavailable(row.timing) ||
                 measured(row.timing, StepClock::Device,
                          StepTimingRelation::NonAdditive))) {
      return false;
    }
  }
  return barriers == profile.execution.pipeline.barrier_count;
}

[[nodiscard]] int CheckFailures(Device &device, Backend backend);
[[nodiscard]] int CheckProfile(Device &device, Backend backend);
[[nodiscard]] int CheckAttribution(Device &device, Backend backend);
[[nodiscard]] int RunTick(Device &device, Backend expected_backend,
                          PipelineProfile profile, Evidence &evidence);

} // namespace package_device_program
