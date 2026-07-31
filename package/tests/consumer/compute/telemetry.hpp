#pragma once

#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <span>

namespace package_compute {

inline int Telemetry() {
  using namespace rund::compute;
  using namespace rund::compute::telemetry;

  auto device = open(Target::cpu(1u));
  if (!device) {
    return device.exit_code();
  }
  const auto identity = device->info();
  auto program = on(*device)
                     .map<std::uint32_t>("profile", 4u,
                                         [](auto value) { return value + 1u; })
                     .compile();
  if (!identity) {
    return identity.exit_code();
  }
  if (!program) {
    return program.exit_code();
  }
  const std::array<std::uint32_t, 4u> input{1u, 2u, 3u, 4u};
  auto job = program->resident(std::span<const std::uint32_t>{input});
  if (!job) {
    return job.exit_code();
  }
  const auto executed = job->run();
  if (!executed) {
    return executed.exit_code();
  }

  const Stats execution = job->stats();
  const MemoryStats memory = job->memory();
  const auto profile = job->profile();
  const auto host_memory = profile ? profile->memory_usage(MemoryCategory::Host)
                                   : Result<Rate>::fail(profile.reason());
  if (!profile) {
    return profile.exit_code();
  }
  if (!host_memory) {
    return host_memory.exit_code();
  }
  const rund::telemetry::Findings findings = profile->findings();
  bool saw_scan = false;
  for (const rund::telemetry::Finding &finding : findings) {
    if (finding.cost == rund::telemetry::Cost::Scan) {
      saw_scan = finding.observed == execution.graph_read_bytes &&
                 finding.cause == rund::telemetry::Cause::GraphRead &&
                 finding.action == rund::telemetry::Action::ReduceGraphBound;
    }
  }
  if (!execution.available() || !memory.available() ||
      profile->device() != *identity ||
      profile->execution().backend != execution.backend ||
      profile->execution().graph_hash != execution.graph_hash ||
      profile->execution().dispatches != execution.dispatches ||
      profile->memory().scope != MemoryScope::Job ||
      profile->kernel_time() !=
          Rate{execution.kernel_ns, execution.kernel_samples} ||
      profile->dispatches_per_submit() !=
          Rate{execution.dispatches, execution.command_submits} ||
      profile->pipeline_cache() !=
          Share{execution.pipeline_cache_hits, execution.pipeline_compiles} ||
      profile->buffer_reuse() !=
          Share{execution.buffer_reuses, execution.buffer_allocations} ||
      profile->descriptor_reuse() !=
          Share{execution.descriptor_reuses,
                execution.descriptor_set_allocations} ||
      profile->dispatch_reduction() !=
          Rate{execution.original_dispatches >= execution.final_dispatches
                   ? execution.original_dispatches - execution.final_dispatches
                   : 0u,
               execution.original_dispatches} ||
      profile->internal_traffic() !=
          Share{execution.internal_roundtrip_bytes,
                execution.external_roundtrip_bytes} ||
      host_memory->numerator != memory.host.current ||
      host_memory->denominator != memory.host.budget || !saw_scan ||
      findings.empty() ||
      findings[findings.size() - 1u].cost !=
          rund::telemetry::Cost::CriticalPath) {
    return 2;
  }
  const Focus focus = profile->largest_time();
  if (focus.includes(Stage::Kernel) && execution.kernel_samples == 0u) {
    return 2;
  }
  const Share saturated{.selected = std::numeric_limits<std::uint64_t>::max(),
                        .other = 1u};
  return Rate{}.value().has_value() || Share{}.value().has_value() ||
                 !saturated.saturated() || saturated.available() ||
                 saturated.value().has_value()
             ? 2
             : 0;
}

} // namespace package_compute
