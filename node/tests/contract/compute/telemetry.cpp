#include <rund/compute.hpp>

#include "../../../src/telemetry/compute.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

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

[[nodiscard]] bool CheckProjection() {
  const rund::compute::Stats stats{
      .backend = rund::compute::Backend::Vulkan,
      .buffer_allocations = 37u,
      .dispatches = 3u,
      .command_submits = 1u,
      .command_capacity = 8u,
      .command_inflight_peak = 8u,
      .uploaded_bytes = 107u,
      .downloaded_bytes = 109u,
      .buffer_reuses = 31u,
      .graph_read_bytes = 113u,
      .kernel_ns = 43u,
      .kernel_samples = 2u,
      .shader_compile_ns = 47u,
      .spirv_compile_ns = 53u,
      .pipeline_create_ns = 59u,
      .descriptor_setup_ns = 61u,
      .submit_wait_ns = 67u,
      .readback_ns = 71u,
      .graph_hash = 73u,
      .worker_count = 5u,
      .participating_workers = 4u,
      .tile_count = 79u,
  };
  const rund::telemetry::Event event = rund::telemetry::detail::ComputeEvent(
      stats, rund::compute::Code::Ok, rund::telemetry::Level::Detail);
  const rund::telemetry::Findings findings = event.findings();
  if (event.level != rund::telemetry::Level::Detail ||
      event.compute.backend != stats.backend ||
      event.compute.graph != stats.graph_hash ||
      event.compute.workers != stats.worker_count ||
      event.compute.active_workers != stats.participating_workers ||
      event.compute.tiles != stats.tile_count ||
      event.compute.copied_bytes != 216u || event.queue.depth != 8u ||
      event.queue.capacity != 8u || event.detail.prepare_ns != 220u ||
      event.detail.work_ns != 67u || event.detail.finish_ns != 71u ||
      findings.size() != rund::telemetry::Findings::Capacity ||
      findings[0u].cost != rund::telemetry::Cost::Allocation ||
      findings[0u].observed != 37u || findings[0u].reference != 31u ||
      findings[1u].cost != rund::telemetry::Cost::Copy ||
      findings[1u].observed != 216u ||
      findings[2u].cost != rund::telemetry::Cost::Scan ||
      findings[2u].observed != 113u ||
      findings[3u].cost != rund::telemetry::Cost::Queue ||
      findings[3u].reference_kind !=
          rund::telemetry::Reference::QueueCapacity ||
      findings[3u].cause != rund::telemetry::Cause::QueueAtBound ||
      findings[4u].cost != rund::telemetry::Cost::CriticalPath ||
      findings[4u].observed != 220u || findings[4u].reference != 358u ||
      findings[4u].cause != rund::telemetry::Cause::Prepare ||
      findings[4u].action != rund::telemetry::Action::ReuseProgram) {
    return false;
  }

  rund::compute::Stats saturated = stats;
  saturated.uploaded_bytes = std::numeric_limits<std::uint64_t>::max();
  saturated.downloaded_bytes = 1u;
  saturated.shader_compile_ns = std::numeric_limits<std::uint64_t>::max();
  const rund::telemetry::Event capped = rund::telemetry::detail::ComputeEvent(
      saturated, rund::compute::Code::Ok, rund::telemetry::Level::Detail);
  const rund::telemetry::Findings capped_findings = capped.findings();
  return capped.compute.copied_bytes ==
             std::numeric_limits<std::uint64_t>::max() &&
         capped.detail.prepare_ns ==
             std::numeric_limits<std::uint64_t>::max() &&
         capped_findings[1u].accuracy == rund::telemetry::Accuracy::Saturated &&
         capped_findings[4u].accuracy == rund::telemetry::Accuracy::Saturated;
}

[[nodiscard]] bool CheckProfile() {
  constexpr std::array<std::uint32_t, 4u> input{1u, 2u, 3u, 4u};
  auto program = rund::compute::on(rund::compute::Target::cpu(1u))
                     .map<std::uint32_t>("telemetry", input.size(),
                                         [](auto value) { return value + 1u; })
                     .compile();
  if (!program) {
    return false;
  }
  auto job = program->resident(input);
  if (!job || !job->run() || !job->read()) {
    return false;
  }
  const auto profile = job->profile();
  if (!profile) {
    return false;
  }
  const rund::telemetry::Findings findings = profile->findings();
  const rund::telemetry::Findings expected =
      rund::telemetry::detail::ComputeFindings(profile->execution());
  if (!Same(findings, expected) || findings.empty() ||
      findings[findings.size() - 1u].cost !=
          rund::telemetry::Cost::CriticalPath ||
      findings[findings.size() - 1u].accuracy !=
          rund::telemetry::Accuracy::Unavailable) {
    return false;
  }
  for (const rund::telemetry::Finding &finding : findings) {
    if (finding.cost == rund::telemetry::Cost::Scan) {
      return finding.observed == profile->execution().graph_read_bytes &&
             finding.cause == rund::telemetry::Cause::GraphRead &&
             finding.action == rund::telemetry::Action::ReduceGraphBound;
    }
  }
  return false;
}

[[nodiscard]] bool CheckSubmissionAction() {
  const rund::compute::Stats stats{
      .backend = rund::compute::Backend::Metal,
      .dispatches = 3u,
      .command_submits = 1u,
      .kernel_ns = 43u,
      .kernel_samples = 1u,
      .submit_wait_ns = 67u,
  };
  const auto event = rund::telemetry::detail::ComputeEvent(
      stats, rund::compute::Code::Ok, rund::telemetry::Level::Detail);
  const auto basic = rund::telemetry::detail::ComputeEvent(
      stats, rund::compute::Code::Ok, rund::telemetry::Level::Basic);
  const auto findings = event.findings();
  if (basic.level != rund::telemetry::Level::Basic ||
      basic.compute.command_submits != 1u || basic.compute.kernel_ns != 0u ||
      basic.compute.kernel_samples != 0u ||
      basic.compute.submit_wait_ns != 0u ||
      event.compute.command_submits != 1u ||
      event.compute.kernel_ns != 43u ||
      event.compute.kernel_samples != 1u ||
      event.compute.submit_wait_ns != 67u || findings.empty()) {
    return false;
  }
  const auto &critical = findings[findings.size() - 1u];
  if (critical.cost != rund::telemetry::Cost::CriticalPath ||
      !rund::telemetry::contains(critical.cause,
                                rund::telemetry::Cause::Work) ||
      !rund::telemetry::contains(critical.cause,
                                rund::telemetry::Cause::SubmitOverhead) ||
      !rund::telemetry::contains(critical.action,
                                rund::telemetry::Action::BatchJobs) ||
      rund::telemetry::contains(critical.action,
                               rund::telemetry::Action::ReduceGraphBound)) {
    return false;
  }

  rund::compute::Stats equal = stats;
  equal.submit_wait_ns = equal.kernel_ns;
  const auto equal_findings = rund::telemetry::detail::ComputeEvent(
                                  equal, rund::compute::Code::Ok,
                                  rund::telemetry::Level::Detail)
                                  .findings();
  const auto &equal_critical = equal_findings[equal_findings.size() - 1u];
  if (rund::telemetry::contains(equal_critical.cause,
                                rund::telemetry::Cause::SubmitOverhead) ||
      rund::telemetry::contains(equal_critical.action,
                                rund::telemetry::Action::BatchJobs) ||
      !rund::telemetry::contains(equal_critical.action,
                                 rund::telemetry::Action::ReduceGraphBound)) {
    return false;
  }

  rund::compute::Stats unsampled = stats;
  unsampled.kernel_samples = 0u;
  const auto unsampled_findings = rund::telemetry::detail::ComputeEvent(
                                      unsampled, rund::compute::Code::Ok,
                                      rund::telemetry::Level::Detail)
                                      .findings();
  const auto &unsampled_critical =
      unsampled_findings[unsampled_findings.size() - 1u];
  return !rund::telemetry::contains(
             unsampled_critical.cause,
             rund::telemetry::Cause::SubmitOverhead) &&
         !rund::telemetry::contains(unsampled_critical.action,
                                    rund::telemetry::Action::BatchJobs) &&
         rund::telemetry::contains(
             unsampled_critical.action,
             rund::telemetry::Action::ReduceGraphBound);
}

} // namespace

int RunComputeTelemetryContract() {
  if (!CheckProjection()) {
    return 1;
  }
  if (!CheckProfile()) {
    return 2;
  }
  return CheckSubmissionAction() ? 0 : 3;
}
