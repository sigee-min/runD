#include "local.hpp"

#include "../allocation.hpp"
#include "src/telemetry/compute.hpp"

#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>

namespace rund_node_test_telemetry {
namespace {

[[nodiscard]] bool Same(const rund::telemetry::Findings &left,
                        const rund::telemetry::Findings &right) noexcept {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0u; index != left.size(); ++index) {
    if (left[index] != right[index]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool SameEpoch(const rund::compute::telemetry::Profile &profile,
                             const rund::compute::Backend backend,
                             const rund::compute::MemoryScope scope) noexcept {
  return profile.device().backend == backend &&
         profile.execution().backend == backend &&
         profile.memory().backend == backend && profile.memory().scope == scope;
}

} // namespace

bool CheckProfiles() {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 4u> input{1u, 2u, 3u, 4u};
  auto device = open(Target::cpu(1u));
  if (!device) {
    return false;
  }
  auto program = on(*device)
                     .map<std::uint32_t>("telemetry-profile", input.size(),
                                         [](auto value) { return value + 1u; })
                     .compile();
  if (!program) {
    return false;
  }
  auto job = program->resident(input);
  if (!job || !job->run() || !job->read()) {
    return false;
  }

  node_compute_allocation::Start();
  const auto job_profile = job->profile();
  node_compute_allocation::Stop();
  if (!job_profile || node_compute_allocation::Count() != 0u ||
      !SameEpoch(*job_profile, Backend::Cpu, MemoryScope::Job)) {
    std::fprintf(
        stderr, "job Profile result=%u reason=%u allocations=%llu\n",
        static_cast<unsigned>(job_profile.ok()),
        static_cast<unsigned>(job_profile.reason()),
        static_cast<unsigned long long>(node_compute_allocation::Count()));
    return false;
  }
  const rund::telemetry::Findings findings = job_profile->findings();
  const rund::telemetry::Findings expected =
      rund::telemetry::detail::ProjectEvent(*job_profile, Code::Ok,
                                            rund::telemetry::Level::Detail)
          .findings();
  if (!Same(findings, expected) || findings.empty() ||
      findings[findings.size() - 1u].cost !=
          rund::telemetry::Cost::CriticalPath) {
    std::fprintf(stderr, "job Profile findings=%zu expected=%zu\n",
                 findings.size(), expected.size());
    return false;
  }

  auto input_buffer = device->upload(std::span<const std::uint32_t>{input});
  auto output_buffer = device->buffer<std::uint32_t>(input.size());
  auto prepared =
      input_buffer && output_buffer
          ? pipeline(*device)
                .then(*program, read(*input_buffer), write(*output_buffer))
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  if (!prepared || !prepared->run()) {
    std::fprintf(stderr, "Pipeline Profile preparation/run failed\n");
    return false;
  }
  node_compute_allocation::Start();
  const auto pipeline_profile = prepared->profile();
  node_compute_allocation::Stop();
  if (!pipeline_profile || node_compute_allocation::Count() != 0u ||
      !SameEpoch(*pipeline_profile, Backend::Cpu, MemoryScope::Pipeline) ||
      pipeline_profile->execution().graph_hash !=
          prepared->stats().graph_hash ||
      pipeline_profile->execution().pipeline.step_count != 1u) {
    std::fprintf(
        stderr,
        "Pipeline Profile result=%u reason=%u allocations=%llu scope=%u "
        "steps=%llu\n",
        static_cast<unsigned>(pipeline_profile.ok()),
        static_cast<unsigned>(pipeline_profile.reason()),
        static_cast<unsigned long long>(node_compute_allocation::Count()),
        pipeline_profile
            ? static_cast<unsigned>(pipeline_profile->memory().scope)
            : 0u,
        pipeline_profile
            ? static_cast<unsigned long long>(
                  pipeline_profile->execution().pipeline.step_count)
            : 0ull);
    return false;
  }
  const DeviceInfo retained_device = pipeline_profile->device();
  prepared = Result<Pipeline>::fail(Reason::PipelineInvalid);
  return retained_device.backend == Backend::Cpu &&
         !retained_device.name.empty() &&
         pipeline_profile->device() == retained_device;
}

} // namespace rund_node_test_telemetry
