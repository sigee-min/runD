#include "../support.hpp"
#include "lifecycle/suite.hpp"

#include <rund/compute.hpp>
#include <rund/compute/async.hpp>
#include <rund/compute/session.hpp>

#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string_view>
#include <thread>

namespace {

[[nodiscard]] auto Program(const std::uint32_t workers) {
  return rund::compute::on(rund::compute::Target::cpu(workers))
      .map<std::int32_t>("node-host-lifecycle", 4u,
                         [](auto value) { return value * 2 + 5; })
      .compile();
}

struct CompletionGate final {
  std::mutex mutex{};
  std::condition_variable changed{};
  bool entered = false;
  bool release = false;

  void operator()(const rund::telemetry::Event &) {
    std::unique_lock lock{mutex};
    entered = true;
    changed.notify_all();
    changed.wait(lock, [&] { return release; });
  }
};

struct CloseObservation final {
  std::mutex mutex{};
  std::condition_variable changed{};
  bool started = false;
  bool returned = false;
  rund::Session::Status status{};
};

} // namespace

int RunRuntimeComputeLifecycleContract() {
  const std::array<std::int32_t, 4> input{1, 2, 3, 4};

  auto program = Program(2u);
  TEST_ASSERT(program);
  auto job = program->resident(input);
  TEST_ASSERT(job);

  rund::Session missing{};
  const auto missing_result = missing.compute(*job).submit().wait();
  TEST_ASSERT(!missing_result);
  TEST_ASSERT(missing_result.error() ==
              std::string_view{"compute_runtime_missing"});

  for (const rund::compute::Compile invalid :
       std::array{rund::compute::Compile{.workers = 0u, .capacity = 1u},
                  rund::compute::Compile{.workers = 1u, .capacity = 0u}}) {
    rund::Session rejected{};
    rund::SessionConfig options = rund::node::test_contract::Options();
    options.compile = invalid;
    const rund::Session::Status status = rejected.open(options);
    TEST_ASSERT(!status);
    TEST_ASSERT(status.code() == rund::ReasonCode::RuntimeResourcesInvalid);
  }

  {
    rund::Session compile_session{};
    rund::SessionConfig options = rund::node::test_contract::Options();
    options.compile = {.workers = 1u, .capacity = 2u};
    TEST_ASSERT(compile_session.open(options));
    auto device = rund::compute::open(
        compile_session, rund::compute::Target::cpu(options.workers));
    TEST_ASSERT(device);
    TEST_ASSERT(device->compile().workers == 1u);
    TEST_ASSERT(device->compile().capacity == 2u);
    auto pending =
        rund::compute::on(*device)
            .map<std::int32_t>("session-compile", 4u,
                               [](auto value) { return value * 2 + 5; })
            .compile_async();
    TEST_ASSERT(pending);
    TEST_ASSERT(pending->get());
    TEST_ASSERT(compile_session.drain());
    auto draining_compile =
        rund::compute::on(*device)
            .map<std::int32_t>("session-compile-draining", 4u,
                               [](auto value) { return value * 2 + 5; })
            .compile_async();
    TEST_ASSERT(!draining_compile);
    TEST_ASSERT(draining_compile.reason() ==
                rund::compute::Reason::AsyncCompileUnavailable);
    TEST_ASSERT(compile_session.close());
    auto stopped_compile =
        rund::compute::on(*device)
            .map<std::int32_t>("session-compile-stopped", 4u,
                               [](auto value) { return value * 2 + 5; })
            .compile_async();
    TEST_ASSERT(!stopped_compile);
    TEST_ASSERT(stopped_compile.reason() ==
                rund::compute::Reason::AsyncCompileUnavailable);
  }

  rund::Session session{};
  TEST_ASSERT(session.open(rund::node::test_contract::Options()).ok());

  auto one_program = Program(1u);
  TEST_ASSERT(one_program);
  auto one_job = one_program->resident(input);
  TEST_ASSERT(one_job);

  const int suite = compute_lifecycle_test::CheckSuite(session, input);
  TEST_ASSERT(suite == 0);
  TEST_ASSERT(rund::node::test_contract::CheckComputeCancelRace(session) == 0);

  const auto mismatch = session.compute(*one_job).submit().wait();
  TEST_ASSERT(!mismatch);
  TEST_ASSERT(mismatch.error() ==
              std::string_view{"compute_node_host_width_mismatch"});
  const auto not_run = one_job->read();
  TEST_ASSERT(!not_run);
  TEST_ASSERT(not_run.error() == std::string_view{"compute_resident_not_run"});
  const rund::Trace disabled_trace = session.trace();
  TEST_ASSERT(!rund::node::test_contract::Saw(
      disabled_trace, rund::TraceEvent::TelemetryEmitted));
  TEST_ASSERT(!rund::node::test_contract::Saw(
      disabled_trace, rund::TraceEvent::TelemetrySkipped));

  TEST_ASSERT(session.drain().ok());
  const auto draining_device = rund::compute::open(
      session, rund::compute::Target::cpu(2u));
  TEST_ASSERT(!draining_device);
  TEST_ASSERT(draining_device.reason() ==
              rund::compute::Reason::RuntimeDraining);
  const auto draining = session.compute(*job).submit().wait();
  TEST_ASSERT(!draining);
  TEST_ASSERT(draining.error() == std::string_view{"compute_runtime_draining"});
  TEST_ASSERT(session.close().ok());
  const auto stopped_device = rund::compute::open(
      session, rund::compute::Target::cpu(2u));
  TEST_ASSERT(!stopped_device);
  TEST_ASSERT(stopped_device.reason() ==
              rund::compute::Reason::RuntimeNotRunning);
  const auto stopped = session.compute(*job).submit().wait();
  TEST_ASSERT(!stopped);
  TEST_ASSERT(stopped.error() == std::string_view{"compute_runtime_missing"});

  {
    auto active_job = program->resident(input);
    auto late_job = program->resident(input);
    TEST_ASSERT(active_job);
    TEST_ASSERT(late_job);
    CompletionGate gate{};
    rund::Session active{};
    rund::SessionConfig active_options = rund::node::test_contract::Options();
    active_options.telemetry = rund::telemetry::bind(gate);
    TEST_ASSERT(active.open(active_options));
    auto active_task = active.compute(*active_job).submit();
    {
      std::unique_lock lock{gate.mutex};
      gate.changed.wait(lock, [&] { return gate.entered; });
    }
    CloseObservation first_close{};
    CloseObservation second_close{};
    std::thread first_closer{[&] {
      {
        std::lock_guard lock{first_close.mutex};
        first_close.started = true;
      }
      first_close.changed.notify_all();
      first_close.status = active.close();
      {
        std::lock_guard lock{first_close.mutex};
        first_close.returned = true;
      }
      first_close.changed.notify_all();
    }};
    std::thread second_closer{[&] {
      {
        std::lock_guard lock{second_close.mutex};
        second_close.started = true;
      }
      second_close.changed.notify_all();
      second_close.status = active.close();
      {
        std::lock_guard lock{second_close.mutex};
        second_close.returned = true;
      }
      second_close.changed.notify_all();
    }};
    {
      std::unique_lock lock{first_close.mutex};
      first_close.changed.wait(lock, [&] { return first_close.started; });
    }
    {
      std::unique_lock lock{second_close.mutex};
      second_close.changed.wait(lock, [&] { return second_close.started; });
    }
    while (active.snapshot().state == rund::SessionState::Running) {
      std::this_thread::yield();
    }
    const rund::Session::Snapshot active_snapshot = active.snapshot();
    TEST_ASSERT(active_snapshot.state == rund::SessionState::Draining);
    TEST_ASSERT(active_snapshot.active_compute_jobs == 1u);
    {
      std::lock_guard lock{first_close.mutex};
      TEST_ASSERT(!first_close.returned);
    }
    {
      std::lock_guard lock{second_close.mutex};
      TEST_ASSERT(!second_close.returned);
    }
    const auto late = active.compute(*late_job).submit().wait();
    TEST_ASSERT(!late);
    TEST_ASSERT(late.error() == std::string_view{"compute_runtime_draining"});
    {
      std::lock_guard lock{gate.mutex};
      gate.release = true;
    }
    gate.changed.notify_all();
    first_closer.join();
    second_closer.join();
    TEST_ASSERT(first_close.status);
    TEST_ASSERT(first_close.status.state() == rund::SessionState::Stopped);
    TEST_ASSERT(second_close.status);
    TEST_ASSERT(second_close.status.state() == rund::SessionState::Stopped);
    TEST_ASSERT(active_task.wait());
    TEST_ASSERT(active.snapshot().state == rund::SessionState::Stopped);
  }

  const int capacity = rund::node::test_contract::CheckComputeTaskCapacity();
  if (capacity != 0) {
    std::fprintf(stderr, "compute task capacity contract failed: %d\n",
                 capacity);
  }
  return capacity;
}
