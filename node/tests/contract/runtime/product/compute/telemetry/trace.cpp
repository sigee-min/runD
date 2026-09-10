#include "local.hpp"

#include "../../support.hpp"
#include "src/runtime/session/result.hpp"

#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/compute/session.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace rund::node::test_contract::telemetry_contract {
namespace {

struct BoundObserver final {
  void operator()(const ::rund::telemetry::Event &) const noexcept {}
};

} // namespace

int CheckTraceAndTiming() {
  constexpr std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto program =
      compute::on(compute::Target::cpu(2u))
          .map<std::int32_t>("node-host-telemetry", input.size(),
                             [](auto value) { return value * 2 + 1; })
          .compile();
  auto trace_job =
      program ? program->resident(input)
              : compute::Result<compute::Job<std::int32_t(std::int32_t)>>::fail(
                    compute::Reason::ProgramInvalid);
  if (!trace_job) {
    return 17;
  }
  TelemetryProbe trace_probe{};
  auto trace_observer = [&](const ::rund::telemetry::Event &event) {
    trace_probe(event);
  };
  ::rund::Session trace_session{};
  rund::SessionConfig trace_options = Options();
  trace_options.telemetry =
      ::rund::telemetry::bind(trace_observer, ::rund::telemetry::Level::Trace);
  if (!trace_session.open(trace_options)) {
    return 18;
  }
  const compute::Completion trace_result =
      trace_session.compute(*trace_job).submit().wait();
  const compute::Stats &trace_stats = trace_result.stats();
  if (!trace_result || trace_probe.events != 1u ||
      trace_probe.event.level != ::rund::telemetry::Level::Trace ||
      trace_stats.dispatches != 1u || trace_stats.kernel_samples != 1u ||
      trace_stats.kernel_ns == 0u ||
      trace_probe.event.compute.kernel_samples != trace_stats.kernel_samples ||
      trace_probe.event.compute.kernel_ns != trace_stats.kernel_ns ||
      trace_probe.event.detail.work_ns != trace_stats.kernel_ns) {
    return 19;
  }

  auto trace_device = compute::open(compute::Target::cpu(2u));
  auto trace_pipeline_program =
      trace_device
          ? compute::on(*trace_device)
                .map<std::int32_t>("node-host-trace-pipeline", input.size(),
                                   [](auto value) { return value * 2 + 1; })
                .compile()
          : compute::Result<compute::Program<std::int32_t(std::int32_t)>>::fail(
                compute::Reason::DeviceInvalid);
  auto trace_input =
      trace_device ? trace_device->upload(std::span<const std::int32_t>{input})
                   : compute::Result<compute::Buffer<std::int32_t>>::fail(
                         compute::Reason::DeviceInvalid);
  auto trace_output =
      trace_device ? trace_device->buffer<std::int32_t>(input.size())
                   : compute::Result<compute::Buffer<std::int32_t>>::fail(
                         compute::Reason::DeviceInvalid);
  auto trace_pipeline =
      trace_device && trace_pipeline_program && trace_input && trace_output
          ? compute::pipeline(*trace_device)
                .then(*trace_pipeline_program, compute::read(*trace_input),
                      compute::write(*trace_output))
                .prepare()
          : compute::Result<compute::Pipeline>::fail(
                compute::Reason::PipelineInvalid);
  if (!trace_pipeline) {
    return 20;
  }
  const compute::Completion trace_pipeline_result =
      trace_session.compute(*trace_pipeline).submit().wait();
  const compute::Stats &trace_pipeline_stats = trace_pipeline_result.stats();
  if (!trace_pipeline_result || trace_probe.events != 2u ||
      trace_probe.event.level != ::rund::telemetry::Level::Trace ||
      trace_pipeline_stats.dispatches != 1u ||
      trace_pipeline_stats.kernel_samples != 1u ||
      trace_pipeline_stats.kernel_ns == 0u ||
      trace_probe.event.compute.kernel_samples !=
          trace_pipeline_stats.kernel_samples ||
      trace_probe.event.compute.kernel_ns != trace_pipeline_stats.kernel_ns) {
    return 21;
  }

  constexpr std::array<std::uint32_t, 4u> scan_input{1u, 2u, 3u, 4u};
  auto trace_scan_program =
      compute::on(compute::Target::cpu(2u))
          .map<std::uint32_t>("node-host-trace-scan", scan_input.size(),
                              [](auto value) { return value; })
          .scan(compute::Scan::InclusiveSum)
          .compile();
  auto trace_scan_job =
      trace_scan_program
          ? trace_scan_program->resident(
                std::span<const std::uint32_t>{scan_input})
          : compute::Result<compute::Job<std::uint32_t(std::uint32_t)>>::fail(
                compute::Reason::ProgramInvalid);
  if (!trace_scan_job) {
    return 22;
  }
  const compute::Completion trace_scan_result =
      trace_session.compute(*trace_scan_job).submit().wait();
  const compute::Stats &trace_scan_stats = trace_scan_result.stats();
  const auto trace_scan_output = trace_scan_job->read();
  if (!trace_scan_result || trace_probe.events != 3u ||
      trace_scan_stats.dispatches < 2u ||
      trace_scan_stats.kernel_samples != trace_scan_stats.dispatches ||
      trace_scan_stats.kernel_ns == 0u || !trace_scan_output ||
      *trace_scan_output != std::vector<std::uint32_t>{1u, 3u, 6u, 10u}) {
    std::fprintf(
        stderr,
        "trace scan ok=%u events=%u dispatches=%llu samples=%llu "
        "kernel_ns=%llu output=%u\n",
        static_cast<unsigned>(trace_scan_result.ok()), trace_probe.events,
        static_cast<unsigned long long>(trace_scan_stats.dispatches),
        static_cast<unsigned long long>(trace_scan_stats.kernel_samples),
        static_cast<unsigned long long>(trace_scan_stats.kernel_ns),
        static_cast<unsigned>(trace_scan_output.ok()));
    return 23;
  }
  const ::rund::Session::Result trace_scope_result = trace_session.scope([] {});
  if (!trace_scope_result ||
      ::rund::detail::session::ResultAccess::telemetry_clock_reads(
          trace_scope_result) != 4u ||
      !trace_session.close()) {
    return 24;
  }

  BoundObserver basic_observer{};
  ::rund::Session basic_scope{};
  rund::SessionConfig basic_options = Options();
  basic_options.telemetry = ::rund::telemetry::bind(basic_observer);
  if (!basic_scope.open(basic_options)) {
    return 25;
  }
  const ::rund::Session::Result basic_result = basic_scope.scope([] {});
  const ::rund::telemetry::Detail &basic_timing =
      ::rund::detail::session::ResultAccess::timing(basic_result);
  if (!basic_result ||
      ::rund::detail::session::ResultAccess::telemetry_clock_reads(
          basic_result) != 0u ||
      basic_timing.prepare_ns != 0u || basic_timing.work_ns != 0u ||
      basic_timing.finish_ns != 0u || !basic_scope.close()) {
    return 26;
  }

  BoundObserver timing_observer{};
  ::rund::Session detail_scope{};
  rund::SessionConfig timing_options = Options();
  timing_options.telemetry = ::rund::telemetry::bind(
      timing_observer, ::rund::telemetry::Level::Detail);
  if (!detail_scope.open(timing_options)) {
    return 27;
  }
  const ::rund::Session::Result timing_result = detail_scope.scope([] {});
  if (!timing_result ||
      ::rund::detail::session::ResultAccess::telemetry_clock_reads(
          timing_result) != 4u ||
      !detail_scope.close()) {
    return 28;
  }
  return 0;
}

} // namespace rund::node::test_contract::telemetry_contract
