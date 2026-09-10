#include "local.hpp"

#include "../../support.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>

#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <thread>

namespace rund::node::test_contract::telemetry_contract {
namespace {

struct TelemetryGate final {
  ::rund::Session *session = nullptr;
  std::mutex mutex{};
  std::condition_variable changed{};
  bool entered = false;
  bool drain_returned = false;
  bool release = false;
  bool reentry_forbidden = false;

  void operator()(const ::rund::telemetry::Event &) {
    const ::rund::Trace trace = session->trace();
    std::unique_lock lock{mutex};
    reentry_forbidden =
        trace.code == ::rund::ReasonCode::RuntimeReentryForbidden;
    entered = true;
    changed.notify_all();
    changed.wait(lock, [&] { return release; });
  }
};

struct ThrowingObserver final {
  void operator()(const ::rund::telemetry::Event &) const { throw 1; }
};

} // namespace

int CheckBasicAndDetail() {
  constexpr std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto program =
      compute::on(compute::Target::cpu(2u))
          .map<std::int32_t>("node-host-telemetry", input.size(),
                             [](auto value) { return value * 2 + 1; })
          .compile();
  if (!program) {
    return 1;
  }
  auto job = program->resident(input);
  if (!job) {
    return 2;
  }

  ::rund::Session session{};
  TelemetryGate gate{.session = &session};
  rund::SessionConfig options = Options();
  options.telemetry = ::rund::telemetry::bind(gate);
  if (!session.open(options)) {
    return 3;
  }
  auto task = session.compute(*job).submit();
  {
    std::unique_lock lock{gate.mutex};
    gate.changed.wait(lock, [&] { return gate.entered; });
  }

  rund::Session::Status drained{};
  rund::Session::Status stopped{};
  std::thread lifecycle{[&] {
    drained = session.drain();
    {
      std::lock_guard lock{gate.mutex};
      gate.drain_returned = true;
    }
    gate.changed.notify_all();
    stopped = session.close();
  }};
  {
    std::unique_lock lock{gate.mutex};
    gate.changed.wait(lock, [&] { return gate.drain_returned; });
    if (session.snapshot().state != ::rund::SessionState::Draining) {
      gate.release = true;
      lock.unlock();
      gate.changed.notify_all();
      lifecycle.join();
      return 4;
    }
    gate.release = true;
  }
  gate.changed.notify_all();
  const compute::Completion result = task.wait();
  lifecycle.join();
  if (!result || result.stats().kernel_ns != 0u ||
      result.stats().kernel_samples != 0u || !drained ||
      !gate.reentry_forbidden) {
    std::fprintf(stderr,
                 "compute telemetry result=%u error=%.*s drained=%u "
                 "drain_reason=%.*s reentry=%u\n",
                 static_cast<bool>(result),
                 static_cast<int>(result.error().size()), result.error().data(),
                 static_cast<bool>(drained),
                 static_cast<int>(drained.error().size()),
                 drained.error().data(), gate.reentry_forbidden);
    return 5;
  }
  if (!stopped && !session.close()) {
    return 6;
  }

  auto throwing_job = program->resident(input);
  if (!throwing_job) {
    return 7;
  }
  ::rund::Session throwing{};
  rund::SessionConfig throwing_options = Options();
  ThrowingObserver throwing_observer{};
  throwing_options.telemetry = ::rund::telemetry::bind(throwing_observer);
  if (!throwing.open(throwing_options)) {
    return 8;
  }
  const compute::Completion throwing_result =
      throwing.compute(*throwing_job).submit().wait();
  if (!throwing_result ||
      !Saw(throwing.trace(), ::rund::TraceEvent::TelemetrySkipped)) {
    return 9;
  }
  if (!throwing.close()) {
    return 10;
  }

  auto detail_job = program->resident(input);
  if (!detail_job) {
    return 11;
  }
  TelemetryProbe detail_probe{};
  auto detail_observer = [&](const ::rund::telemetry::Event &event) {
    detail_probe(event);
  };
  ::rund::Session detail_session{};
  rund::SessionConfig detail_options = Options();
  detail_options.telemetry = ::rund::telemetry::bind(
      detail_observer, ::rund::telemetry::Level::Detail);
  if (!detail_session.open(detail_options)) {
    return 12;
  }
  const compute::Completion detail_result =
      detail_session.compute(*detail_job).submit().wait();
  if (!detail_result || detail_probe.events != 1u ||
      detail_probe.event.source != ::rund::telemetry::Source::Compute ||
      detail_probe.event.level != ::rund::telemetry::Level::Detail ||
      detail_probe.event.compute.code != compute::Code::Ok ||
      !detail_probe.event.error().empty() ||
      detail_probe.event.replay.code != ::rund::replay::Code::Ok) {
    return 13;
  }
  const compute::Stats &detail_stats = detail_result.stats();
  const std::uint64_t expected_prepare =
      detail_stats.shader_compile_ns + detail_stats.spirv_compile_ns +
      detail_stats.pipeline_create_ns + detail_stats.descriptor_setup_ns;
  const std::uint64_t expected_work = detail_stats.submit_wait_ns != 0u
                                          ? detail_stats.submit_wait_ns
                                          : detail_stats.kernel_ns;
  if (detail_probe.event.detail.prepare_ns != expected_prepare ||
      detail_probe.event.detail.work_ns != expected_work ||
      detail_probe.event.detail.finish_ns != detail_stats.readback_ns ||
      detail_probe.event.compute.graph_read_bytes !=
          detail_stats.graph_read_bytes ||
      detail_probe.event.compute.graph_read_bytes != sizeof(input) ||
      detail_probe.event.replay.mode != ::rund::telemetry::Mode::None ||
      detail_probe.event.replay.plan != ::rund::telemetry::Preparation::None ||
      detail_probe.event.replay.input_rows != 0u ||
      detail_probe.event.replay.input_bytes != 0u ||
      detail_probe.event.replay.choices != 0u ||
      detail_probe.event.replay.evidence_rows != 0u ||
      detail_probe.event.replay.evidence_bytes != 0u ||
      detail_probe.event.replay.retained_bytes != 0u ||
      detail_probe.event.replay.copied_bytes != 0u ||
      detail_probe.event.replay.physical_bytes != 0u ||
      detail_probe.event.replay.allocated_bytes != 0u ||
      detail_probe.event.replay.reserved_bytes != 0u ||
      detail_probe.event.replay.storage_growths != 0u ||
      detail_probe.event.replay.result_hash != 0u) {
    return 14;
  }
  const auto basic_output = job->read();
  const auto detail_output = detail_job->read();
  if (!basic_output || !detail_output || *basic_output != *detail_output) {
    return 15;
  }
  return detail_session.close() ? 0 : 16;
}

} // namespace rund::node::test_contract::telemetry_contract
