#include "support.hpp"

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

namespace {

struct ReasonSchemaRow {
  std::uint16_t value;
  rund::ReasonCode code;
  std::string_view cpp_name;
  std::string_view text;
  bool prepared_memory;
};

constexpr std::array reason_schema{
#define RUND_NODE_REASON_CATEGORY_Universal true
#define RUND_NODE_REASON_CATEGORY_General false
#define RUND_NODE_REASON_CATEGORY_PreparedMemory true
#define RUND_NODE_REASON(value, name, text, category)                          \
  ReasonSchemaRow{value, rund::ReasonCode::name, #name, text,                  \
                  RUND_NODE_REASON_CATEGORY_##category},
#include <rund/reason.def>
#undef RUND_NODE_REASON
#undef RUND_NODE_REASON_CATEGORY_PreparedMemory
#undef RUND_NODE_REASON_CATEGORY_General
#undef RUND_NODE_REASON_CATEGORY_Universal
};

constexpr bool valid_reason_schema() {
  if (reason_schema.size() != 102u || reason_schema.front().value != 0u ||
      reason_schema.back().value != 101u) {
    return false;
  }
  std::size_t prepared_memory_count = 0u;
  for (std::size_t index = 0u; index < reason_schema.size(); ++index) {
    const ReasonSchemaRow &row = reason_schema[index];
    if (static_cast<std::uint16_t>(row.code) != row.value ||
        row.value != index || row.cpp_name.empty() || row.text.empty()) {
      return false;
    }
    prepared_memory_count += row.prepared_memory ? 1u : 0u;
  }
  return prepared_memory_count == 11u;
}

static_assert(valid_reason_schema());
static_assert(
    std::is_same_v<std::underlying_type_t<rund::ReasonCode>, std::uint16_t>);
static_assert(sizeof(rund::ReasonCode) == sizeof(std::uint16_t));

template <class Result> constexpr bool has_product_outcome_contract() {
  return std::is_member_function_pointer_v<decltype(&Result::ok)> &&
         noexcept(std::declval<const Result &>().ok()) &&
         noexcept(static_cast<bool>(std::declval<const Result &>())) &&
         noexcept(std::declval<const Result &>().error()) &&
         noexcept(std::declval<const Result &>().exit_code());
}

static_assert(has_product_outcome_contract<rund::Session::Status>());
static_assert(has_product_outcome_contract<rund::Session::Result>());
static_assert(has_product_outcome_contract<rund::Session::Snapshot>());
static_assert(
    std::is_same_v<decltype(rund::Session::Result{}.code()), rund::ReasonCode>);
static_assert(
    std::is_same_v<decltype(rund::Session::Status{}.code()), rund::ReasonCode>);
static_assert(std::is_same_v<decltype(rund::Session::Status{}.state()),
                             rund::SessionState>);
static_assert(!std::is_copy_constructible_v<rund::Session::Result>);
static_assert(!std::is_copy_assignable_v<rund::Session::Result>);
static_assert(std::is_nothrow_move_constructible_v<rund::Session::Result>);
static_assert(std::is_nothrow_move_assignable_v<rund::Session::Result>);
static_assert(!std::is_constructible_v<rund::Session::Status, rund::ReasonCode,
                                       rund::SessionState>);

constexpr rund::Session::Status failed_decision{};
static_assert(!failed_decision.ok());
static_assert(!static_cast<bool>(failed_decision));
static_assert(failed_decision.exit_code() == 1);

struct SchedulerReentry final {
  rund::Session::Status opened{};
  rund::Session::Status drained{};
  rund::Session::Status closed{};
  rund::Session::Snapshot snapshot{};
  rund::Trace trace{};
  rund::Resources resources{};
  rund::ReasonCode scope = rund::ReasonCode::Ok;
};

void ObserveSchedulerReentry(rund::Session &session, SchedulerReentry &probe) {
  probe.snapshot = session.snapshot();
  probe.trace = session.trace();
  probe.resources = session.resources();
  probe.scope = session.scope([] {}).code();
  probe.opened = session.open(rund::node::test_contract::Options());
  probe.drained = session.drain();
  probe.closed = session.close();
}

rund::task::Task<void>
ObserveSchedulerReentryAfterYield(rund::Session *const session,
                                  SchedulerReentry *const probe,
                                  rund::task::Status *const yielded) {
  *yielded = co_await rund::task::yield();
  ObserveSchedulerReentry(*session, *probe);
}

void AssertSchedulerReentry(const SchedulerReentry &probe) {
  TEST_ASSERT(!probe.opened);
  TEST_ASSERT(probe.opened.code() == rund::ReasonCode::RuntimeReentryForbidden);
  TEST_ASSERT(probe.opened.state() == rund::SessionState::Running);
  TEST_ASSERT(!probe.drained);
  TEST_ASSERT(probe.drained.code() ==
              rund::ReasonCode::RuntimeReentryForbidden);
  TEST_ASSERT(probe.drained.state() == rund::SessionState::Running);
  TEST_ASSERT(!probe.closed);
  TEST_ASSERT(probe.closed.code() == rund::ReasonCode::RuntimeReentryForbidden);
  TEST_ASSERT(probe.closed.state() == rund::SessionState::Running);
  TEST_ASSERT(!probe.snapshot);
  TEST_ASSERT(probe.snapshot.code == rund::ReasonCode::RuntimeReentryForbidden);
  TEST_ASSERT(probe.snapshot.state == rund::SessionState::Running);
  TEST_ASSERT(probe.trace.code == rund::ReasonCode::RuntimeReentryForbidden);
  TEST_ASSERT(probe.trace.records.empty());
  TEST_ASSERT(!probe.resources);
  TEST_ASSERT(probe.resources.code ==
              rund::ReasonCode::RuntimeReentryForbidden);
  TEST_ASSERT(probe.scope == rund::ReasonCode::RuntimeReentryForbidden);
}

} // namespace

int RunRuntimeLifecycleContract() {
  using namespace rund::node::test_contract;

  for (std::uint16_t value = 0u; value < reason_schema.size(); ++value) {
    const auto code = static_cast<rund::ReasonCode>(value);
    TEST_ASSERT(rund::ValidReasonCode(code));
    TEST_ASSERT(reason_schema[value].code == code);
    TEST_ASSERT(rund::ValidPreparedMemoryReason(code) ==
                reason_schema[value].prepared_memory);
    TEST_ASSERT(std::string_view{rund::ReasonString(code)} ==
                reason_schema[value].text);
  }
  constexpr std::array<std::uint16_t, 2u> unknown{
      static_cast<std::uint16_t>(reason_schema.size()),
      std::numeric_limits<std::uint16_t>::max()};
  for (const std::uint16_t value : unknown) {
    const auto code = static_cast<rund::ReasonCode>(value);
    TEST_ASSERT(!rund::ValidReasonCode(code));
    TEST_ASSERT(!rund::ValidPreparedMemoryReason(code));
    TEST_ASSERT(std::string_view{rund::ReasonString(code)} ==
                "unknown_reason_code");
  }

  rund::Session missing_id{};
  const rund::Session::Status missing =
      missing_id.open(rund::SessionConfig{.id = 0u});
  TEST_ASSERT(!missing);
  TEST_ASSERT(missing.error() == std::string_view{"runtime_id_required"});
  TEST_ASSERT(failed_decision.error() == std::string_view{"not_configured"});
  const rund::Session::Status unopened_close = missing_id.close();
  TEST_ASSERT(!unopened_close);
  TEST_ASSERT(unopened_close.code() == rund::ReasonCode::NotConfigured);
  TEST_ASSERT(unopened_close.state() == rund::SessionState::Unconfigured);

  rund::Session invalid_telemetry{};
  rund::SessionConfig invalid_telemetry_options = Options();
  auto ignore_telemetry = [](const rund::telemetry::Event &) {};
  invalid_telemetry_options.telemetry = rund::telemetry::bind(
      ignore_telemetry, static_cast<rund::telemetry::Level>(
                            std::numeric_limits<std::uint8_t>::max()));
  const rund::Session::Status invalid_level =
      invalid_telemetry.open(invalid_telemetry_options);
  TEST_ASSERT(!invalid_level);
  TEST_ASSERT(invalid_level.error() ==
              std::string_view{"telemetry_level_invalid"});

  rund::Session runtime{};
  const rund::Session::Status opened = runtime.open(Options());
  TEST_ASSERT(opened);
  TEST_ASSERT(opened.error().empty());
  TEST_ASSERT(opened.exit_code() == 0);
  TEST_ASSERT(opened.code() == rund::ReasonCode::Ok);
  TEST_ASSERT(opened.state() == rund::SessionState::Running);
  const rund::Session::Status duplicate_open = runtime.open(Options());
  TEST_ASSERT(!duplicate_open);
  TEST_ASSERT(duplicate_open.code() == rund::ReasonCode::AlreadyConfigured);
  TEST_ASSERT(duplicate_open.state() == rund::SessionState::Running);
  TEST_ASSERT(runtime.snapshot().active_compute_jobs == 0u);
  TEST_ASSERT(!runtime.snapshot().scope_active);
  TEST_ASSERT(runtime.snapshot().ok());
  TEST_ASSERT(runtime.snapshot().error().empty());
  TEST_ASSERT(runtime.snapshot().state == rund::SessionState::Running);
  rund::Session::Snapshot nested{};
  const rund::Session::Result nested_scope =
      runtime.scope([&] { nested = runtime.snapshot(); });
  TEST_ASSERT(nested_scope);
  TEST_ASSERT(!nested);
  TEST_ASSERT(nested.code == rund::ReasonCode::RuntimeReentryForbidden);
  TEST_ASSERT(nested.state == rund::SessionState::Running);
  TEST_ASSERT(nested.error() == std::string_view{"runtime_reentry_forbidden"});
  TEST_ASSERT(nested.exit_code() == 1);
  rund::Session::Status reentrant_open{};
  rund::Session::Status reentrant_close{};
  const rund::Session::Result closing_scope = runtime.scope([&] {
    reentrant_open = runtime.open(Options());
    reentrant_close = runtime.close();
  });
  TEST_ASSERT(closing_scope);
  TEST_ASSERT(!reentrant_open);
  TEST_ASSERT(reentrant_open.code() ==
              rund::ReasonCode::RuntimeReentryForbidden);
  TEST_ASSERT(reentrant_open.state() == rund::SessionState::Running);
  TEST_ASSERT(!reentrant_close);
  TEST_ASSERT(reentrant_close.code() ==
              rund::ReasonCode::RuntimeReentryForbidden);
  TEST_ASSERT(reentrant_close.state() == rund::SessionState::Running);
  TEST_ASSERT(runtime.snapshot().state == rund::SessionState::Running);
  const rund::Session::Result scope = runtime.scope([] {});
  TEST_ASSERT(scope);
  TEST_ASSERT(scope.scope() != 0u);
  TEST_ASSERT(!runtime.snapshot().scope_active);
  const rund::Session::Result next_scope = runtime.scope([] {});
  TEST_ASSERT(next_scope);
  TEST_ASSERT(next_scope.scope() > scope.scope());

  SchedulerReentry leaf_reentry{};
  SchedulerReentry coroutine_reentry{};
  rund::task::Status leaf_join{};
  rund::task::Status coroutine_join{};
  rund::task::Status coroutine_yield{};
  rund::Session independent{};
  rund::SessionConfig independent_options = Options();
  independent_options.id = 2u;
  TEST_ASSERT(independent.open(independent_options));
  rund::Session::Snapshot independent_snapshot{};
  rund::Session::Status independent_close{};
  rund::task::Status independent_join{};
  const rund::Session::Result scheduler_reentry = runtime.scope([&] {
    const rund::task::Handle leaf =
        rund::task::spawn("session-reentry-leaf", [&] {
          ObserveSchedulerReentry(runtime, leaf_reentry);
        });
    leaf_join = rund::task::join(leaf);

    const rund::task::Handle coroutine =
        rund::task::spawn("session-reentry-coroutine",
                          ObserveSchedulerReentryAfterYield(
                              &runtime, &coroutine_reentry, &coroutine_yield));
    coroutine_join = rund::task::join(coroutine);

    const rund::task::Handle independent_task =
        rund::task::spawn("session-independent-lifecycle", [&] {
          independent_snapshot = independent.snapshot();
          independent_close = independent.close();
        });
    independent_join = rund::task::join(independent_task);
  });
  TEST_ASSERT(scheduler_reentry);
  TEST_ASSERT(leaf_join);
  TEST_ASSERT(coroutine_join);
  TEST_ASSERT(coroutine_yield);
  TEST_ASSERT(independent_join);
  AssertSchedulerReentry(leaf_reentry);
  AssertSchedulerReentry(coroutine_reentry);
  TEST_ASSERT(independent_snapshot);
  TEST_ASSERT(independent_snapshot.state == rund::SessionState::Running);
  TEST_ASSERT(independent_close);
  TEST_ASSERT(independent_close.state() == rund::SessionState::Stopped);
  TEST_ASSERT(runtime.snapshot().state == rund::SessionState::Running);

  const rund::Session::Status closed = runtime.close();
  TEST_ASSERT(closed);
  TEST_ASSERT(closed.state() == rund::SessionState::Stopped);
  TEST_ASSERT(runtime.snapshot().state == rund::SessionState::Stopped);
  const rund::Session::Status repeated_close = runtime.close();
  TEST_ASSERT(repeated_close);
  TEST_ASSERT(repeated_close.state() == rund::SessionState::Stopped);
  const rund::Session::Status stopped_drain = runtime.drain();
  TEST_ASSERT(!stopped_drain);
  TEST_ASSERT(stopped_drain.code() == rund::ReasonCode::NotRunnable);
  TEST_ASSERT(stopped_drain.state() == rund::SessionState::Stopped);

  rund::Session deliberate{};
  TEST_ASSERT(deliberate.open(Options()));
  const rund::Session::Status draining = deliberate.drain();
  TEST_ASSERT(draining);
  TEST_ASSERT(draining.state() == rund::SessionState::Draining);
  const rund::Session::Status repeated_drain = deliberate.drain();
  TEST_ASSERT(!repeated_drain);
  TEST_ASSERT(repeated_drain.code() == rund::ReasonCode::NotRunnable);
  TEST_ASSERT(repeated_drain.state() == rund::SessionState::Draining);
  TEST_ASSERT(deliberate.close());

  rund::Session active{};
  TEST_ASSERT(active.open(Options()));
  std::mutex gate_mutex{};
  std::condition_variable gate_changed{};
  bool entered = false;
  bool try_reentry = false;
  bool reentered = false;
  bool release = false;
  std::size_t close_started = 0u;
  std::size_t close_returned = 0u;
  rund::Session::Status callback_close{};
  rund::Session::Status first_close{};
  rund::Session::Status second_close{};
  rund::ReasonCode active_scope_code = rund::ReasonCode::SessionResultMissing;
  std::thread worker{[&] {
    const rund::Session::Result result = active.scope([&] {
      {
        std::lock_guard lock{gate_mutex};
        entered = true;
      }
      gate_changed.notify_all();
      std::unique_lock lock{gate_mutex};
      gate_changed.wait(lock, [&] { return try_reentry; });
      lock.unlock();
      callback_close = active.close();
      lock.lock();
      reentered = true;
      gate_changed.notify_all();
      gate_changed.wait(lock, [&] { return release; });
    });
    active_scope_code = result.code();
  }};
  {
    std::unique_lock lock{gate_mutex};
    gate_changed.wait(lock, [&] { return entered; });
  }
  const auto close_active = [&](rund::Session::Status &status) {
    {
      std::lock_guard lock{gate_mutex};
      ++close_started;
    }
    gate_changed.notify_all();
    status = active.close();
    {
      std::lock_guard lock{gate_mutex};
      ++close_returned;
    }
    gate_changed.notify_all();
  };
  std::thread first_closer{[&] { close_active(first_close); }};
  std::thread second_closer{[&] { close_active(second_close); }};
  {
    std::unique_lock lock{gate_mutex};
    gate_changed.wait(lock, [&] { return close_started == 2u; });
  }
  while (active.snapshot().state == rund::SessionState::Running) {
    std::this_thread::yield();
  }
  TEST_ASSERT(active.snapshot().state == rund::SessionState::Draining);
  {
    std::lock_guard lock{gate_mutex};
    TEST_ASSERT(close_returned == 0u);
  }
  const rund::Session::Result late_scope = active.scope([] {});
  TEST_ASSERT(!late_scope);
  TEST_ASSERT(late_scope.code() == rund::ReasonCode::RuntimeScopeNotStarted);
  {
    std::lock_guard lock{gate_mutex};
    try_reentry = true;
  }
  gate_changed.notify_all();
  {
    std::unique_lock lock{gate_mutex};
    gate_changed.wait(lock, [&] { return reentered; });
    TEST_ASSERT(close_returned == 0u);
  }
  TEST_ASSERT(!callback_close);
  TEST_ASSERT(callback_close.code() ==
              rund::ReasonCode::RuntimeReentryForbidden);
  TEST_ASSERT(callback_close.state() == rund::SessionState::Draining);
  {
    std::lock_guard lock{gate_mutex};
    release = true;
  }
  gate_changed.notify_all();
  worker.join();
  first_closer.join();
  second_closer.join();
  TEST_ASSERT(active_scope_code == rund::ReasonCode::Ok);
  TEST_ASSERT(first_close);
  TEST_ASSERT(first_close.state() == rund::SessionState::Stopped);
  TEST_ASSERT(second_close);
  TEST_ASSERT(second_close.state() == rund::SessionState::Stopped);
  TEST_ASSERT(active.snapshot().state == rund::SessionState::Stopped);
  const rund::Trace active_trace = active.trace();
  std::uint32_t draining_records = 0u;
  std::uint32_t stopped_records = 0u;
  std::uint64_t draining_sequence = 0u;
  std::uint64_t stopped_sequence = 0u;
  for (const rund::TraceRecord &record : active_trace.records) {
    if (record.event == rund::TraceEvent::RuntimeDraining) {
      ++draining_records;
      draining_sequence = record.sequence;
    } else if (record.event == rund::TraceEvent::RuntimeStopped) {
      ++stopped_records;
      stopped_sequence = record.sequence;
    }
  }
  TEST_ASSERT(draining_records == 1u);
  TEST_ASSERT(stopped_records == 1u);
  TEST_ASSERT(draining_sequence < stopped_sequence);

  rund::Session trace_runtime{};
  rund::SessionConfig trace_options = Options();
  trace_options.trace_capacity = 3u;
  TEST_ASSERT(trace_runtime.open(trace_options));
  TEST_ASSERT(trace_runtime.close());
  TEST_ASSERT(trace_runtime.close());
  const rund::Trace trace = trace_runtime.trace();
  TEST_ASSERT(trace.dropped == 1u);
  TEST_ASSERT(trace.records.size() == 3u);
  TEST_ASSERT(trace.records[0].event == rund::TraceEvent::RuntimeStarted);
  TEST_ASSERT(trace.records[1].event == rund::TraceEvent::RuntimeDraining);
  TEST_ASSERT(trace.records[2].event == rund::TraceEvent::RuntimeStopped);
  TEST_ASSERT(trace.records[0].sequence == 2u);
  TEST_ASSERT(trace.records[1].sequence == 3u);
  TEST_ASSERT(trace.records[2].sequence == 4u);
  return 0;
}
