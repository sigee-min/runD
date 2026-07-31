#include "test/assert.hpp"

#include <rund/session.hpp>
#include <rund/task/cancel.hpp>

namespace {

[[nodiscard]] rund::SessionConfig TaskCancelRunSpec() noexcept {
  return rund::SessionConfig{
    .workers = 1u,
    .scheduler = {
      .task_workers = 1u,
      .task_capacity = 4u,
      .ready_queue_capacity = 4u,
      .timer_capacity = 4u,
      .reactor_wait_capacity = 4u,
      .observation_capacity = 32u,
      .host_event_capacity = 32u,
    },
  };
}

}  // namespace

int RunRuntimeTaskCancelContract() {
  bool source_valid = false;
  bool token_valid = false;
  bool before_stop_state_ok = false;
  bool before_stop_requested = true;
  bool first_stop_ok = false;
  bool after_stop_state_ok = false;
  bool after_stop_requested = false;
  bool second_stop_ok = false;
  rund::task::stop_source stale_source{};
  rund::task::Status stale_cross_scope_stop{};
  bool new_scope_source_valid = false;
  bool new_scope_token_valid = false;
  bool new_scope_state_ok = false;
  bool new_scope_requested_after_stale = true;
  bool new_scope_stop_ok = false;

  const rund::Session::Result report = rund::run(TaskCancelRunSpec(), [&] {
    auto source = rund::task::stop_source::create();
    stale_source = source;
    source_valid = static_cast<bool>(source);
    if (!source_valid) {
      return;
    }

    auto token = source.token();
    rund::task::stop_token explicit_token = token;
    token_valid = static_cast<bool>(explicit_token);
    if (!token_valid) {
      return;
    }

    const rund::task::StopState before_stop = explicit_token.state();
    before_stop_state_ok = before_stop.ok();
    before_stop_requested = before_stop.requested();
    first_stop_ok = source.request_stop().ok();
    const rund::task::StopState after_stop = explicit_token.state();
    after_stop_state_ok = after_stop.ok();
    after_stop_requested = after_stop.requested();
    second_stop_ok = source.request_stop().ok();
  });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(source_valid);
  TEST_ASSERT(token_valid);
  TEST_ASSERT(before_stop_state_ok);
  TEST_ASSERT(!before_stop_requested);
  TEST_ASSERT(first_stop_ok);
  TEST_ASSERT(after_stop_state_ok);
  TEST_ASSERT(after_stop_requested);
  TEST_ASSERT(second_stop_ok);

  const rund::Session::Result stale_report = rund::run(TaskCancelRunSpec(), [&] {
    auto source = rund::task::stop_source::create();
    new_scope_source_valid = static_cast<bool>(source);
    if (!new_scope_source_valid) {
      return;
    }
    auto token = source.token();
    new_scope_token_valid = static_cast<bool>(token);
    if (!new_scope_token_valid) {
      return;
    }

    stale_cross_scope_stop = stale_source.request_stop();
    const rund::task::StopState new_scope_state = token.state();
    new_scope_state_ok = new_scope_state.ok();
    new_scope_requested_after_stale = new_scope_state.requested();
    new_scope_stop_ok = source.request_stop().ok();
  });

  TEST_ASSERT(stale_report.ok());
  TEST_ASSERT(new_scope_source_valid);
  TEST_ASSERT(new_scope_token_valid);
  TEST_ASSERT(!stale_cross_scope_stop.ok());
  TEST_ASSERT(stale_cross_scope_stop.code() ==
              rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(new_scope_state_ok);
  TEST_ASSERT(!new_scope_requested_after_stale);
  TEST_ASSERT(new_scope_stop_ok);
  return 0;
}
