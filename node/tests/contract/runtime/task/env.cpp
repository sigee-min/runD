#include "coroutine/allocation.hpp"
#include "src/runtime/replay/scope/session.hpp"
#include "src/runtime/replay/scope/timing.hpp"
#include "src/runtime/session/result.hpp"
#include "src/runtime/task/scheduler/state/storage.hpp"
#include "test/assert.hpp"

#include <rund/host/env.hpp>
#include <rund/host/event.hpp>
#include <rund/host/hash.hpp>
#include <rund/session.hpp>
#include <rund/replay.hpp>
#include <rund/host.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

#include <unistd.h>

namespace {

using EnvResult = rund::task::Result<std::string>;

[[nodiscard]] EnvResult PendingEnvResult() {
  return EnvResult::fail(rund::ReasonCode::TaskInvalid);
}

struct EventScript final {
  std::array<rund::host::Event, 3u> events{};
  bool ok = true;
  bool lease_valid = false;
};

void RunEventScript(void *const raw, rund::Session &,
                    const rund::replay::detail::scope::Lease lease) {
  auto &script = *static_cast<EventScript *>(raw);
  script.lease_valid = lease.valid();
  rund::node::Scheduler *const scheduler = rund::node::Scheduler::Active();
  script.ok = scheduler != nullptr;
  for (const rund::host::Event &event : script.events) {
    script.ok =
        scheduler != nullptr && scheduler->RecordHostEvent(event) && script.ok;
  }
}

} // namespace

int RunRuntimeTaskEnvContract() {
  EnvResult path_result = PendingEnvResult();
  EnvResult missing_result = PendingEnvResult();
  const std::string missing_name =
      std::string{"RUND_ENV_CONTRACT_MISSING_"} +
      std::to_string(static_cast<long long>(::getpid()));

  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
                  .host_event_capacity = 4u,
              },
      },
      [&] {
        path_result = rund::host::env::get("PATH");
        missing_result = rund::host::env::get(missing_name);
      });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(path_result);
  TEST_ASSERT(path_result.error().empty());
  TEST_ASSERT(path_result.code() == rund::ReasonCode::Ok);
  TEST_ASSERT(missing_result);
  TEST_ASSERT(missing_result.error().empty());
  TEST_ASSERT(missing_result.code() == rund::ReasonCode::Ok);
  TEST_ASSERT(missing_result->empty());
  TEST_ASSERT(report.tasks().host_events() >= 2u);
  TEST_ASSERT(report.events().size() >= 2u);

  const rund::host::Event &path_event = report.events()[0];
  TEST_ASSERT(path_event.kind == rund::host::EventKind::EnvGet);
  TEST_ASSERT(path_event.status == rund::host::Status::Ok);
  TEST_ASSERT(path_event.name_hash.value ==
              rund::host::hash_string("PATH", 4u).value);
  TEST_ASSERT(
      path_event.payload_hash.value ==
      rund::host::hash_string(path_result->data(), path_result->size())
          .value);

  const rund::host::Event &missing_event = report.events()[1];
  TEST_ASSERT(missing_event.kind == rund::host::EventKind::EnvGet);
  TEST_ASSERT(missing_event.status == rund::host::Status::Ok);
  TEST_ASSERT(
      missing_event.name_hash.value ==
      rund::host::hash_string(missing_name.data(), missing_name.size())
          .value);
  TEST_ASSERT(missing_event.payload_hash.value ==
              rund::host::hash_string("", 0u).value);

  rund::Session repeated{};
  rund::SessionConfig repeated_config{};
  repeated_config.id = 41u;
  repeated_config.workers = 1u;
  repeated_config.scheduler.host_event_capacity = 2u;
  TEST_ASSERT(repeated.open(repeated_config));
  bool first_recorded = false;
  rund::task::Status first_join{};
  const rund::Session::Result first_scope = repeated.scope([&] {
    first_join = rund::task::join(rund::task::spawn("repeated-env", [] {}));
    rund::node::Scheduler *const scheduler = rund::node::Scheduler::Active();
    first_recorded = scheduler != nullptr &&
                     scheduler->RecordHostEvent(
                         {.kind = rund::host::EventKind::EnvGet});
  });
  TEST_ASSERT(first_scope);
  TEST_ASSERT(first_join);
  TEST_ASSERT(first_recorded);
  TEST_ASSERT(first_scope.events().size() == 1u);
  const rund::host::Event frozen_event = first_scope.events()[0];

  bool second_recorded = false;
  rund::task::Status second_join{};
  std::uint64_t second_allocations = ~std::uint64_t{0u};
  const rund::Session::Result second_scope = repeated.scope([&] {
    runtime_task_allocation::Start();
    second_join = rund::task::join(rund::task::spawn("repeated-env", [] {}));
    rund::node::Scheduler *const scheduler = rund::node::Scheduler::Active();
    second_recorded = scheduler != nullptr &&
                      scheduler->RecordHostEvent(
                          {.kind = rund::host::EventKind::EnvGet});
    runtime_task_allocation::Stop();
    second_allocations = runtime_task_allocation::Count();
  });
  TEST_ASSERT(second_scope);
  TEST_ASSERT(second_join);
  TEST_ASSERT(second_recorded);
  TEST_ASSERT(second_allocations == 0u);
  TEST_ASSERT(second_scope.events().size() == 1u);
  TEST_ASSERT(second_scope.events()[0].sequence == 1u);
  TEST_ASSERT(second_scope.tasks().host_events() == 1u);
  TEST_ASSERT(first_scope.tasks().trace_hash() ==
              second_scope.tasks().trace_hash());
  TEST_ASSERT(first_scope.events().size() == 1u);
  TEST_ASSERT(first_scope.events()[0].sequence == frozen_event.sequence);
  TEST_ASSERT(first_scope.events()[0].name_hash.value ==
              frozen_event.name_hash.value);
  TEST_ASSERT(first_scope.tasks().spawned() == 1u);
  TEST_ASSERT(first_scope.tasks().completed() == 1u);
  TEST_ASSERT(first_scope.tasks().task_record_allocations() == 1u);
  TEST_ASSERT(first_scope.tasks().task_record_reuses() == 0u);
  TEST_ASSERT(second_scope.tasks().spawned() == 1u);
  TEST_ASSERT(second_scope.tasks().completed() == 1u);
  TEST_ASSERT(second_scope.tasks().task_record_allocations() == 0u);
  TEST_ASSERT(second_scope.tasks().task_record_reuses() == 1u);
  TEST_ASSERT(repeated.close());

  {
    rund::node::SchedulerPlanState projection{};
    TEST_ASSERT(projection.task(7u) == 7u);
    TEST_ASSERT(projection.handle(8u) == 8u);
    TEST_ASSERT(projection.descriptor(9) == 9);
    TEST_ASSERT(projection.failure == rund::ReasonCode::Ok);
    projection.installed = true;
    projection.bases.task = 9u;
    TEST_ASSERT(projection.task(0u) == 0u);
    TEST_ASSERT(projection.task(9u) == 1u);
    TEST_ASSERT(projection.task(12u) == 4u);
    TEST_ASSERT(projection.failure == rund::ReasonCode::Ok);
    TEST_ASSERT(projection.task(8u) == 0u);
    TEST_ASSERT(projection.failure ==
                rund::ReasonCode::ReplayScopeIdentityInvalid);

    projection.failure = rund::ReasonCode::Ok;
    projection.bases.task = 0u;
    TEST_ASSERT(projection.task(std::numeric_limits<std::uint64_t>::max()) ==
                0u);
    TEST_ASSERT(projection.failure ==
                rund::ReasonCode::ReplayScopeIdentityInvalid);
  }

  {
    rund::node::SchedulerHandleMap defaults{};
    const bool configured =
        defaults.configure(rund::SchedulerConfig{}.host_handle_capacity);
    TEST_ASSERT(configured);
    TEST_ASSERT(defaults.entries.size() == 2048u);
    TEST_ASSERT(sizeof(rund::node::SchedulerHandleMap::Entry) == 24u);
    TEST_ASSERT(defaults.entries.size() *
                    sizeof(rund::node::SchedulerHandleMap::Entry) ==
                49152u);
  }

  {
    rund::node::SchedulerPlanState handles{};
    const bool configured = handles.configure_handles(2u);
    TEST_ASSERT(configured);
    handles.installed = true;
    runtime_task_allocation::Start();
    handles.begin();
    TEST_ASSERT(handles.handle(41u) == 1u);
    TEST_ASSERT(handles.handle(42u) == 2u);
    TEST_ASSERT(handles.handle(41u) == 1u);
    handles.retire(41u);
    TEST_ASSERT(handles.handle(41u) == 3u);
    runtime_task_allocation::Stop();
    TEST_ASSERT(runtime_task_allocation::Count() == 0u);
    TEST_ASSERT(handles.failure == rund::ReasonCode::Ok);
    TEST_ASSERT(handles.handle(43u) == 0u);
    TEST_ASSERT(handles.failure ==
                rund::ReasonCode::ReplayScopeIdentityInvalid);
  }

  {
    rund::node::SchedulerPlanState overflow{};
    TEST_ASSERT(overflow.configure_handles(1u));
    overflow.installed = true;
    overflow.begin();
    overflow.handles.next = std::numeric_limits<std::uint64_t>::max();
    TEST_ASSERT(overflow.handle(51u) ==
                std::numeric_limits<std::uint64_t>::max());
    overflow.retire(51u);
    TEST_ASSERT(overflow.handle(52u) == 0u);
    TEST_ASSERT(overflow.failure ==
                rund::ReasonCode::ReplayScopeIdentityInvalid);
  }

  rund::Session identity_session{};
  rund::SessionConfig identity_config{};
  identity_config.workers = 1u;
  identity_config.scheduler.host_event_capacity = 4u;
  identity_config.scheduler.host_handle_capacity = 4u;
  TEST_ASSERT(identity_session.open(identity_config));
  EventScript recorded_script{
      .events =
          {
              rund::host::Event{
                  .kind = rund::host::EventKind::NetSocket,
                  .status = rund::host::Status::Ok,
                  .host_handle_id = 41u},
              rund::host::Event{.kind =
                                          rund::host::EventKind::IoClose,
                                      .status = rund::host::Status::Ok,
                                      .host_handle_id = 41u},
              rund::host::Event{
                  .kind = rund::host::EventKind::NetSocket,
                  .status = rund::host::Status::Ok,
                  .host_handle_id = 41u},
          },
  };
  rund::replay::detail::scope::Timing record_timing{false};
  rund::Session::Result recorded_scope =
      rund::replay::detail::scope::Access::run(
          identity_session,
          rund::replay::detail::scope::Plan{
              .mode = rund::replay::detail::scope::Mode::Record},
          &recorded_script, RunEventScript, record_timing);
  TEST_ASSERT(recorded_scope);
  TEST_ASSERT(recorded_script.ok);
  TEST_ASSERT(recorded_script.lease_valid);
  TEST_ASSERT(recorded_scope.events().size() == 3u);
  TEST_ASSERT(recorded_scope.events()[0].host_handle_id == 1u);
  TEST_ASSERT(recorded_scope.events()[1].host_handle_id == 1u);
  TEST_ASSERT(recorded_scope.events()[2].host_handle_id == 2u);

  auto prepared = rund::replay::detail::scope::Access::prepare(
      identity_session,
      rund::detail::session::ResultAccess::take_events(recorded_scope),
      rund::detail::session::ResultAccess::take_payloads(recorded_scope));
  TEST_ASSERT(prepared);
  EventScript replayed_script = recorded_script;
  for (rund::host::Event &event : replayed_script.events) {
    event.host_handle_id = 501u;
  }
  rund::replay::detail::scope::Timing replay_timing{false};
  const rund::Session::Result replayed_scope =
      rund::replay::detail::scope::Access::run(
          identity_session,
          rund::replay::detail::scope::Plan{
              .mode = rund::replay::detail::scope::Mode::Replay,
              .expected = prepared.owner},
          &replayed_script, RunEventScript, replay_timing);
  TEST_ASSERT(replayed_scope);
  TEST_ASSERT(replayed_script.ok);
  TEST_ASSERT(replayed_script.lease_valid);
  TEST_ASSERT(replayed_scope.events().size() == 3u);
  TEST_ASSERT(replayed_scope.events()[0].host_handle_id == 1u);
  TEST_ASSERT(replayed_scope.events()[1].host_handle_id == 1u);
  TEST_ASSERT(replayed_scope.events()[2].host_handle_id == 2u);
  TEST_ASSERT(recorded_scope.tasks().trace_hash() ==
              replayed_scope.tasks().trace_hash());
  TEST_ASSERT(identity_session.close());

  rund::Session bounded_identity{};
  rund::SessionConfig bounded_config{};
  bounded_config.workers = 1u;
  bounded_config.scheduler.host_event_capacity = 4u;
  bounded_config.scheduler.host_handle_capacity = 1u;
  TEST_ASSERT(bounded_identity.open(bounded_config));
  bool first_identity = false;
  bool second_identity = true;
  bool third_identity = true;
  const rund::Session::Result bounded_scope = bounded_identity.scope([&] {
    rund::node::Scheduler *const scheduler = rund::node::Scheduler::Active();
    first_identity = scheduler != nullptr &&
                     scheduler->RecordHostEvent(
                         {.kind = rund::host::EventKind::NetSocket,
                          .status = rund::host::Status::Ok,
                          .host_handle_id = 71u});
    second_identity = scheduler != nullptr &&
                      scheduler->RecordHostEvent(
                          {.kind = rund::host::EventKind::NetSocket,
                           .status = rund::host::Status::Ok,
                           .host_handle_id = 72u});
    third_identity = scheduler != nullptr &&
                     scheduler->RecordHostEvent(
                         {.kind = rund::host::EventKind::NetSocket,
                          .status = rund::host::Status::Ok,
                          .host_handle_id = 73u});
  });
  TEST_ASSERT(!bounded_scope);
  TEST_ASSERT(bounded_scope.code() ==
              rund::ReasonCode::ReplayScopeIdentityInvalid);
  TEST_ASSERT(first_identity);
  TEST_ASSERT(!second_identity);
  TEST_ASSERT(!third_identity);
  TEST_ASSERT(bounded_scope.events().size() == 1u);
  TEST_ASSERT(bounded_scope.tasks().host_events() == 1u);
  TEST_ASSERT(bounded_identity.close());

  rund::Session disabled_identity{};
  rund::SessionConfig disabled_config{};
  disabled_config.workers = 1u;
  disabled_config.scheduler.host_event_capacity = 1u;
  disabled_config.scheduler.host_handle_capacity = 0u;
  TEST_ASSERT(disabled_identity.open(disabled_config));
  bool disabled_recorded = true;
  const rund::Session::Result disabled_scope = disabled_identity.scope([&] {
    rund::node::Scheduler *const scheduler = rund::node::Scheduler::Active();
    disabled_recorded =
        scheduler != nullptr &&
        scheduler->RecordHostEvent({.kind = rund::host::EventKind::IoRead,
                                    .status = rund::host::Status::Ok,
                                    .host_handle_id = 81u});
  });
  TEST_ASSERT(!disabled_scope);
  TEST_ASSERT(disabled_scope.code() ==
              rund::ReasonCode::ReplayScopeIdentityInvalid);
  TEST_ASSERT(!disabled_recorded);
  TEST_ASSERT(disabled_scope.events().empty());
  TEST_ASSERT(disabled_scope.tasks().host_events() == 0u);
  TEST_ASSERT(disabled_identity.close());

  EnvResult spawned_path_result = PendingEnvResult();
  rund::task::Status spawned_join{};
  const rund::Session::Result spawned_report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
                  .host_event_capacity = 4u,
              },
      },
      [&] {
        const rund::task::Handle task = rund::task::spawn("spawned-env", [&] {
          spawned_path_result = rund::host::env::get("PATH");
        });
        spawned_join = rund::task::join(task);
      });
  TEST_ASSERT(spawned_report.ok());
  TEST_ASSERT(spawned_join.ok());
  TEST_ASSERT(spawned_path_result);
  TEST_ASSERT(spawned_report.tasks().host_events() >= 1u);
  TEST_ASSERT(!spawned_report.events().empty());
  TEST_ASSERT(spawned_report.events().front().kind ==
              rund::host::EventKind::EnvGet);
  TEST_ASSERT(spawned_report.events().front().status ==
              rund::host::Status::Ok);

  constexpr std::size_t kLaneEnvTasks = 4u;
  constexpr std::array<std::string_view, kLaneEnvTasks> kLaneEnvNames{
      "PATH", "HOME", "SHELL", "RUND_MISSING_LANE_ENV"};
  std::array<EnvResult, kLaneEnvTasks> lane_results{
      PendingEnvResult(),
      PendingEnvResult(),
      PendingEnvResult(),
      PendingEnvResult(),
  };
  rund::task::Status lane_join{};
  const rund::Session::Result lane_report = rund::run(
      rund::SessionConfig{
          .workers = 8u,
          .scheduler =
              {
                  .task_workers = 8u,
                  .task_capacity = 8u,
                  .ready_queue_capacity = 8u,
                  .host_event_capacity = 8u,
              },
      },
      [&] {
        std::array<rund::task::Handle, kLaneEnvTasks> tasks{};
        for (std::size_t index = 0u; index < tasks.size(); ++index) {
          tasks[index] = rund::task::spawn("lane-env", [&, index] {
            lane_results[index] = rund::host::env::get(kLaneEnvNames[index]);
          });
        }
        lane_join = rund::task::join(tasks[0], tasks[1], tasks[2], tasks[3]);
      });
  TEST_ASSERT(lane_report.ok());
  TEST_ASSERT(lane_join.ok());
  TEST_ASSERT(lane_report.events().size() == kLaneEnvTasks);
  for (const EnvResult &lane_result : lane_results) {
    TEST_ASSERT(lane_result);
  }
  for (std::size_t index = 0u; index < lane_report.events().size(); ++index) {
    const rund::host::Event &event = lane_report.events()[index];
    TEST_ASSERT(event.kind == rund::host::EventKind::EnvGet);
    TEST_ASSERT(event.status == rund::host::Status::Ok);
    TEST_ASSERT(event.name_hash.value ==
                rund::host::hash_string(kLaneEnvNames[index].data(),
                                        kLaneEnvNames[index].size())
                    .value);
  }

  const EnvResult invalid = rund::host::env::get("");
  TEST_ASSERT(!invalid);
  TEST_ASSERT(invalid.error() == "task_invalid");
  TEST_ASSERT(invalid.code() == rund::ReasonCode::TaskInvalid);

  const std::string embedded_nul_name{"PATH\0SUFFIX", 11u};
  const EnvResult embedded_nul = rund::host::env::get(
      std::string_view{embedded_nul_name.data(), embedded_nul_name.size()});
  TEST_ASSERT(!embedded_nul);
  TEST_ASSERT(embedded_nul.error() == "task_invalid");
  TEST_ASSERT(embedded_nul.code() == rund::ReasonCode::TaskInvalid);

  return 0;
}
