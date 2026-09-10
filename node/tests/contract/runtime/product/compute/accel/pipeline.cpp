#include "../../support.hpp"
#include "pipeline/local.hpp"

#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/compute/session.hpp>

#include "src/accel/kernel/fault.hpp"
#include "src/compute/device/state.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>

namespace {

using runtime_compute_pipeline_accel_detail::CheckFixedRecurrences;
using runtime_compute_pipeline_accel_detail::CheckPipelineControl;

int Check(const rund::compute::Target target, rund::compute::Device &device) {
  constexpr std::array<std::int32_t, 4u> input_values{1, 2, 3, 4};
  auto first = rund::compute::on(device)
                   .map<std::int32_t>("session-accel-pipeline-double",
                                      input_values.size(),
                                      [](auto value) { return value * 2; })
                   .compile();
  auto second = rund::compute::on(device)
                    .map<std::int32_t>("session-accel-pipeline-advance",
                                       input_values.size(),
                                       [](auto value) { return value + 3; })
                    .compile();
  auto input = device.upload<std::int32_t>(input_values);
  auto middle = device.buffer<std::int32_t>(input_values.size());
  auto output = device.buffer<std::int32_t>(input_values.size());
  if (!first || !second || !input || !middle || !output) {
    return 2;
  }
  auto prepared = rund::compute::pipeline(device)
                      .then(*first, rund::compute::read(*input),
                            rund::compute::write(*middle))
                      .then(*second, rund::compute::read(*middle),
                            rund::compute::write(*output))
                      .prepare();
  if (!prepared) {
    std::fprintf(stderr, "pipeline prepare backend=%u reason=%.*s\n",
                 static_cast<unsigned>(target.backend()),
                 static_cast<int>(prepared.error().size()),
                 prepared.error().data());
    return 3;
  }
  rund::Session session{};
  if (!session.open(rund::node::test_contract::Options())) {
    return 4;
  }
  auto submission = session.compute(*prepared).submit();
  const rund::compute::Completion completion = submission.wait();
  if (!completion) {
    return 5;
  }
  const rund::compute::Stats stats = completion.stats();
  if (stats.backend != target.backend() || stats.pipeline.step_count != 2u ||
      stats.pipeline.resource_count != 3u ||
      stats.pipeline.barrier_count != 1u ||
      stats.pipeline.verified_step_count != 2u ||
      stats.pipeline.failed_step_index !=
          rund::compute::PipelineStats::no_failed_step ||
      stats.pipeline.control_byte_count != 128u ||
      stats.pipeline.control_command_count != 2u ||
      stats.command_submits != 1u || stats.dispatches != 2u ||
      prepared->generation() != 1u) {
    std::fprintf(
        stderr,
        "pipeline topology backend=%u steps=%llu resources=%llu barriers=%llu "
        "verified=%llu failed=%llu control-bytes=%llu control-commands=%llu "
        "submits=%llu dispatches=%llu generation=%llu\n",
        static_cast<unsigned>(target.backend()),
        static_cast<unsigned long long>(stats.pipeline.step_count),
        static_cast<unsigned long long>(stats.pipeline.resource_count),
        static_cast<unsigned long long>(stats.pipeline.barrier_count),
        static_cast<unsigned long long>(stats.pipeline.verified_step_count),
        static_cast<unsigned long long>(stats.pipeline.failed_step_index),
        static_cast<unsigned long long>(stats.pipeline.control_byte_count),
        static_cast<unsigned long long>(stats.pipeline.control_command_count),
        static_cast<unsigned long long>(stats.command_submits),
        static_cast<unsigned long long>(stats.dispatches),
        static_cast<unsigned long long>(prepared->generation()));
    return 6;
  }
  std::array<std::int32_t, input_values.size()> values{};
  if (!prepared->read(*output, values) ||
      values != std::array<std::int32_t, 4u>{5, 7, 9, 11}) {
    return 7;
  }
  rund::node::test_contract::TelemetryProbe trace_probe{};
  rund::Session trace_session{};
  rund::SessionConfig trace_options = rund::node::test_contract::Options();
  trace_options.telemetry =
      ::rund::telemetry::bind(trace_probe, ::rund::telemetry::Level::Trace);
  if (!trace_session.open(trace_options)) {
    return 9;
  }
  const std::shared_ptr<rund::compute::detail::DeviceState> &device_state =
      rund::compute::detail::DeviceAccess::state(device);
  rund::compute::detail::AccelDeviceState *const native =
      device_state == nullptr
          ? nullptr
          : rund::compute::detail::accel_device(*device_state);
  const rund::compute::MemoryStats before_trace_memory = prepared->memory();
  if (native == nullptr ||
      !rund::node::accel::detail::InjectNativeTraceUnavailableOnce(
          native->pick)) {
    return 9;
  }
  const rund::compute::Completion unavailable_trace =
      trace_session.compute(*prepared).submit().wait();
  const rund::compute::Stats unavailable_stats = unavailable_trace.stats();
  const rund::compute::MemoryStats unavailable_memory = prepared->memory();
  if (unavailable_trace ||
      unavailable_trace.reason() !=
          rund::compute::Reason::TelemetryTraceUnavailable ||
      unavailable_stats.command_submits != 0u ||
      unavailable_stats.buffer_allocations != 0u ||
      prepared->generation() != 1u ||
      unavailable_memory.host.current != before_trace_memory.host.current ||
      unavailable_memory.host.cumulative !=
          before_trace_memory.host.cumulative ||
      unavailable_memory.device.current != before_trace_memory.device.current ||
      unavailable_memory.device.cumulative !=
          before_trace_memory.device.cumulative) {
    return 9;
  }
  const rund::compute::Completion trace_completion =
      trace_session.compute(*prepared).submit().wait();
  const rund::compute::Stats trace_stats = trace_completion.stats();
  const rund::compute::MemoryStats cold_trace_memory = prepared->memory();
  const std::uint64_t trace_submits =
      target.backend() == rund::compute::Backend::Metal ? 2u : 1u;
  if (!trace_completion || trace_stats.command_submits != trace_submits ||
      trace_stats.dispatches != stats.dispatches ||
      trace_stats.kernel_samples != trace_stats.dispatches ||
      trace_stats.kernel_ns == 0u ||
      trace_stats.graph_hash != stats.graph_hash || trace_probe.events != 2u ||
      (cold_trace_memory.host.current == before_trace_memory.host.current &&
       cold_trace_memory.device.current ==
           before_trace_memory.device.current)) {
    std::fprintf(stderr,
                 "pipeline trace backend=%u ok=%u submits=%llu dispatches=%llu "
                 "samples=%llu ns=%llu events=%u reason=%u\n",
                 static_cast<unsigned>(target.backend()),
                 static_cast<unsigned>(!!trace_completion),
                 static_cast<unsigned long long>(trace_stats.command_submits),
                 static_cast<unsigned long long>(trace_stats.dispatches),
                 static_cast<unsigned long long>(trace_stats.kernel_samples),
                 static_cast<unsigned long long>(trace_stats.kernel_ns),
                 trace_probe.events,
                 static_cast<unsigned>(trace_completion.reason()));
    return 10;
  }
  const rund::compute::Completion warm_trace_completion =
      trace_session.compute(*prepared).submit().wait();
  const rund::compute::Stats warm_trace_stats = warm_trace_completion.stats();
  if (!warm_trace_completion || warm_trace_stats.buffer_allocations != 0u ||
      warm_trace_stats.command_submits != trace_submits ||
      warm_trace_stats.dispatches != trace_stats.dispatches ||
      warm_trace_stats.kernel_samples != warm_trace_stats.dispatches ||
      warm_trace_stats.graph_hash != trace_stats.graph_hash ||
      trace_probe.events != 3u) {
    return 11;
  }
  if (const int control = CheckPipelineControl(session, device); control != 0) {
    return control;
  }
  if (const int fixed = CheckFixedRecurrences(device); fixed != 0) {
    return fixed;
  }
  // A device-loss terminal may invalidate backend-global native state. Keep
  // that destructive fault last so this contract never treats post-loss
  // compilation as a supported retry surface.
  const bool faulted =
      target.backend() == rund::compute::Backend::Metal
          ? rund::node::accel::detail::InjectNativeTraceResolveDeviceLostOnce(
                native->pick)
          : rund::node::accel::detail::InjectNativeDeviceLostOnce(native->pick);
  if (!faulted) {
    return 12;
  }
  const rund::compute::Completion lost_trace =
      trace_session.compute(*prepared).submit().wait();
  const std::uint64_t lost_submits =
      target.backend() == rund::compute::Backend::Metal ? 2u : 1u;
  const rund::compute::Completion poisoned_trace =
      trace_session.compute(*prepared).submit().wait();
  if (lost_trace || lost_trace.reason() != rund::compute::Reason::DeviceLost ||
      lost_trace.stats().command_submits != lost_submits ||
      lost_trace.stats().kernel_samples != 0u || !prepared->poisoned() ||
      prepared->generation() != 3u || trace_probe.events != 4u ||
      poisoned_trace.reason() != rund::compute::Reason::DeviceLost ||
      !trace_session.close()) {
    std::fprintf(
        stderr,
        "pipeline trace loss backend=%u ok=%u reason=%u submits=%llu "
        "samples=%llu poisoned=%u generation=%llu events=%u retry=%u\n",
        static_cast<unsigned>(target.backend()),
        static_cast<unsigned>(!!lost_trace),
        static_cast<unsigned>(lost_trace.reason()),
        static_cast<unsigned long long>(lost_trace.stats().command_submits),
        static_cast<unsigned long long>(lost_trace.stats().kernel_samples),
        static_cast<unsigned>(prepared->poisoned()),
        static_cast<unsigned long long>(prepared->generation()),
        trace_probe.events, static_cast<unsigned>(poisoned_trace.reason()));
    return 12;
  }
  return session.close() ? 0 : 8;
}

} // namespace

int RunRuntimeComputePipelineAccelContract() {
  for (const rund::compute::Target target :
       {rund::compute::Target::metal(), rund::compute::Target::vulkan()}) {
    auto device = rund::compute::open(target);
    if (!device) {
      if (device.reason() != rund::compute::Reason::AdapterUnavailable) {
        return static_cast<int>(target.backend()) * 10 + 1;
      }
      continue;
    }
    const int result = Check(target, *device);
    if (result != 0) {
      return static_cast<int>(target.backend()) * 10 + result;
    }
  }
  return 0;
}
