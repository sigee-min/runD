#pragma once

#include <rund/compute.hpp>

#include <array>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <utility>

namespace package_compute {

template <class Selector>
concept AcceptsOpenSelector =
    requires(Selector selector) { rund::compute::open(selector); };

template <class Selector>
concept AcceptsFlowSelector =
    requires(Selector selector) { rund::compute::on(selector); };

static_assert(!std::constructible_from<rund::compute::Target,
                                       rund::compute::Backend, std::uint32_t>);
static_assert(!AcceptsOpenSelector<rund::compute::Backend>);
static_assert(!AcceptsFlowSelector<rund::compute::Backend>);

[[nodiscard]] inline int MemoryMismatch(const int line) {
  std::fprintf(stderr, "package memory mismatch at line %d\n", line);
  return 2;
}

inline bool Zero(const rund::compute::MemoryCounter counter) {
  return counter.current == 0u && counter.peak == 0u &&
         counter.cumulative == 0u && counter.reused == 0u &&
         counter.budget == 0u;
}

inline bool Unspecified(const rund::compute::MemoryStats &stats) {
  return !stats.available() &&
         stats.backend == rund::compute::Backend::Unavailable &&
         stats.scope == rund::compute::MemoryScope::Unspecified &&
         Zero(stats.host) && Zero(stats.frame) && Zero(stats.tile) &&
         Zero(stats.resident) && Zero(stats.staging) && Zero(stats.device) &&
         Zero(stats.transfer);
}

inline int Memory() {
  static_assert(rund::compute::MemoryStats{}.scope ==
                rund::compute::MemoryScope::Unspecified);
  static_assert(!rund::compute::MemoryStats{}.available());
  static_assert(rund::compute::MemoryStats{}.backend ==
                rund::compute::Backend::Unavailable);
  static_assert(rund::compute::MemorySnapshot{}.summary.scope ==
                rund::compute::MemoryScope::Unspecified);

  const std::array<std::int32_t, 4u> input{1, 2, 3, 4};
  auto device = rund::compute::open(rund::compute::Target::cpu(1u));
  if (!device) {
    return device.exit_code();
  }
  if (device->memory().scope != rund::compute::MemoryScope::Backend) {
    return MemoryMismatch(__LINE__);
  }
  auto owned_device = std::move(*device);
  if (!Unspecified(device->memory()) ||
      owned_device.memory().scope != rund::compute::MemoryScope::Backend) {
    return MemoryMismatch(__LINE__);
  }

  auto program = rund::compute::on(owned_device)
                     .map<std::int32_t>("memory", input.size(),
                                        [](auto value) { return value; })
                     .compile();
  if (!program) {
    return program.exit_code();
  }
  if (program->memory().scope != rund::compute::MemoryScope::Program) {
    return MemoryMismatch(__LINE__);
  }
  auto owned_program = std::move(*program);
  std::array<rund::compute::MemoryEntry, 1u> invalid_entries{};
  const auto invalid_snapshot = program->memory_snapshot(invalid_entries);
  if (!Unspecified(invalid_snapshot.summary) ||
      invalid_snapshot.written != 0u || invalid_snapshot.total != 0u ||
      invalid_snapshot.truncated() ||
      owned_program.memory().scope != rund::compute::MemoryScope::Program) {
    return MemoryMismatch(__LINE__);
  }
  auto job = owned_program.resident(input);
  if (!job) {
    return job.exit_code();
  }
  if (!job->memory().available() ||
      job->memory().scope != rund::compute::MemoryScope::Job ||
      job->memory().resident.current !=
          input.size() * sizeof(std::int32_t) * 3u) {
    return MemoryMismatch(__LINE__);
  }
  auto owned_job = std::move(*job);
  if (!Unspecified(job->memory()) || !owned_job.memory().available() ||
      owned_job.memory().scope != rund::compute::MemoryScope::Job ||
      job->stats().available() ||
      job->stats().backend != rund::compute::Backend::Unavailable) {
    return MemoryMismatch(__LINE__);
  }
  std::array<rund::compute::MemoryEntry, 2u> entries{};
  const auto snapshot = owned_job.memory_snapshot(entries);
  if (!snapshot.truncated() || snapshot.written != entries.size() ||
      snapshot.total <= snapshot.written) {
    return MemoryMismatch(__LINE__);
  }
  return 0;
}

} // namespace package_compute
