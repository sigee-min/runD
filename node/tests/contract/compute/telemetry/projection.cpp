#include "local.hpp"

#include "../allocation.hpp"
#include "src/compute/memory/profile.hpp"
#include "src/telemetry/compute.hpp"

#include <rund/compute.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

namespace rund_node_test_telemetry {
namespace {

[[nodiscard]] rund::compute::telemetry::Profile
Profile(const rund::compute::Stats stats) {
  auto device = std::make_shared<const rund::compute::DeviceInfo>(
      rund::compute::DeviceInfo{.backend = stats.backend,
                                .name = "telemetry-test",
                                .driver = "contract",
                                .driver_details = "projection"});
  return rund::compute::detail::ProfileAccess::make(
      std::move(device), stats,
      rund::compute::MemoryStats{
          .backend = stats.backend,
          .scope = rund::compute::MemoryScope::Job,
      });
}

static_assert(std::is_trivially_copyable_v<rund::telemetry::Event>);
static_assert(std::is_standard_layout_v<rund::telemetry::Event>);
static_assert(static_cast<std::uint8_t>(rund::telemetry::Level::Basic) == 0u);
static_assert(static_cast<std::uint8_t>(rund::telemetry::Level::Detail) == 1u);
static_assert(static_cast<std::uint8_t>(rund::telemetry::Level::Trace) == 2u);
static_assert(rund::compute::detail::valid(
    rund::compute::Reason::TelemetryTraceUnavailable));
static_assert(rund::compute::detail::category(
                  rund::compute::Reason::TelemetryTraceUnavailable) ==
              rund::compute::Code::Unavailable);

} // namespace

bool CheckProjection() {
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
      .transfer_submissions = {.host_to_device = 2u,
                               .device_to_host = 3u,
                               .device_to_device = 5u},
      .host_write_bytes = 127u,
  };
  const auto profile = Profile(stats);
  node_compute_allocation::Start();
  const rund::telemetry::Event event = rund::telemetry::detail::ProjectEvent(
      profile, rund::compute::Code::Ok, rund::telemetry::Level::Detail);
  const rund::telemetry::Event basic = rund::telemetry::detail::ProjectEvent(
      profile, rund::compute::Code::Ok, rund::telemetry::Level::Basic);
  const rund::telemetry::Event trace = rund::telemetry::detail::ProjectEvent(
      profile, rund::compute::Code::Ok, rund::telemetry::Level::Trace);
  const rund::telemetry::Findings findings = event.findings();
  node_compute_allocation::Stop();
  if (node_compute_allocation::Count() != 0u ||
      event.level != rund::telemetry::Level::Detail ||
      basic.level != rund::telemetry::Level::Basic ||
      trace.level != rund::telemetry::Level::Trace ||
      basic.compute.kernel_ns != 0u || basic.compute.kernel_samples != 0u ||
      basic.compute.submit_wait_ns != 0u || basic.detail.prepare_ns != 0u ||
      basic.detail.work_ns != 0u || basic.detail.finish_ns != 0u ||
      basic.compute.host_to_device_submits != 2u ||
      basic.compute.device_to_host_submits != 3u ||
      basic.compute.device_to_device_submits != 5u ||
      event.compute.backend != stats.backend ||
      event.compute.graph != stats.graph_hash ||
      event.compute.workers != stats.worker_count ||
      event.compute.active_workers != stats.participating_workers ||
      event.compute.tiles != stats.tile_count ||
      event.compute.host_to_device_submits != 2u ||
      event.compute.device_to_host_submits != 3u ||
      event.compute.device_to_device_submits != 5u ||
      trace.compute.kernel_ns != stats.kernel_ns ||
      trace.compute.kernel_samples != stats.kernel_samples ||
      trace.compute.submit_wait_ns != stats.submit_wait_ns ||
      trace.detail.prepare_ns != 220u || trace.detail.work_ns != 67u ||
      trace.detail.finish_ns != 71u || event.compute.copied_bytes != 216u ||
      event.queue.depth != 8u || event.queue.capacity != 8u ||
      event.detail.prepare_ns != 220u || event.detail.work_ns != 67u ||
      event.detail.finish_ns != 71u ||
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
  const auto saturated_profile = Profile(saturated);
  const rund::telemetry::Event capped = rund::telemetry::detail::ProjectEvent(
      saturated_profile, rund::compute::Code::Ok,
      rund::telemetry::Level::Detail);
  const rund::telemetry::Findings capped_findings = capped.findings();
  return capped.compute.copied_bytes ==
             std::numeric_limits<std::uint64_t>::max() &&
         capped.detail.prepare_ns ==
             std::numeric_limits<std::uint64_t>::max() &&
         capped_findings[1u].accuracy == rund::telemetry::Accuracy::Saturated &&
         capped_findings[4u].accuracy == rund::telemetry::Accuracy::Saturated;
}

bool CheckSubmissionAction() {
  const rund::compute::Stats stats{
      .backend = rund::compute::Backend::Metal,
      .dispatches = 3u,
      .command_submits = 1u,
      .kernel_ns = 43u,
      .kernel_samples = 1u,
      .submit_wait_ns = 67u,
  };
  const auto profile = Profile(stats);
  const auto event = rund::telemetry::detail::ProjectEvent(
      profile, rund::compute::Code::Ok, rund::telemetry::Level::Detail);
  const auto findings = event.findings();
  if (event.compute.command_submits != 1u || event.compute.kernel_ns != 43u ||
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
  const auto equal_profile = Profile(equal);
  const auto equal_findings = rund::telemetry::detail::ProjectEvent(
                                  equal_profile, rund::compute::Code::Ok,
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
  const auto unsampled_profile = Profile(unsampled);
  const auto unsampled_findings =
      rund::telemetry::detail::ProjectEvent(unsampled_profile,
                                            rund::compute::Code::Ok,
                                            rund::telemetry::Level::Detail)
          .findings();
  const auto &unsampled_critical =
      unsampled_findings[unsampled_findings.size() - 1u];
  return !rund::telemetry::contains(unsampled_critical.cause,
                                    rund::telemetry::Cause::SubmitOverhead) &&
         !rund::telemetry::contains(unsampled_critical.action,
                                    rund::telemetry::Action::BatchJobs) &&
         rund::telemetry::contains(unsampled_critical.action,
                                   rund::telemetry::Action::ReduceGraphBound);
}

} // namespace rund_node_test_telemetry
