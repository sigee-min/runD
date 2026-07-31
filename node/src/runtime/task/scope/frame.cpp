#include "../../replay/scope/session.hpp"
#include "../../replay/scope/timing.hpp"
#include "../../runtime/local.hpp"
#include "../../session/result.hpp"
#include "../scheduler/access.hpp"
#include "../scheduler/state.hpp"

#include "evidence.hpp"
#include <node/runtime/runtime.hpp>
#include <rund/session/memory.hpp>
#include <rund/task/api/access.hpp>
#include <rund/task/await/access.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <utility>

namespace rund::detail::session {

::rund::Session::Result
ResultAccess::make(const ReasonCode code, const std::uint64_t scope,
                   ::rund::node::ScopeEvidence evidence, ::rund::Trace trace,
                   const ::rund::telemetry::Detail detail,
                   const std::uint8_t telemetry_clock_reads) {
  ::rund::Session::Result result{code,
                                 scope,
                                 std::move(evidence.tasks),
                                 std::move(evidence.memory),
                                 std::move(evidence.observations),
                                 std::move(evidence.events),
                                 evidence.input_rows,
                                 evidence.input_bytes,
                                 evidence.ready_capacity,
                                 std::move(trace),
                                 detail,
                                 telemetry_clock_reads};
  *static_cast<::rund::node::replay_detail::payload::Archive *>(
      result.storage()) = std::move(evidence.payloads);
  return result;
}

::rund::Session::Result ResultAccess::fail(const ReasonCode code,
                                           ::rund::Trace trace) {
  return make(code, 0u, {}, std::move(trace));
}

} // namespace rund::detail::session

namespace rund::node {

::rund::Session::Result Runtime::capture_result(TaskScopeFrame &scope,
                                                const ReasonCode code,
                                                ::rund::Trace trace) {
  return ::rund::detail::session::ResultAccess::make(
      code, scope.id(), scope.evidence(), std::move(trace));
}

Runtime::TraceCursor Runtime::begin_trace() const noexcept {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return TraceCursor{.next = state_->next_trace,
                     .dropped = state_->dropped_trace};
}

::rund::Trace Runtime::capture_trace(const TraceCursor begin) const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  ::rund::Trace result{};
  const std::uint64_t generated = state_->trace_capacity == 0u
                                      ? state_->dropped_trace - begin.dropped
                                      : state_->next_trace - begin.next;
  result.records.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(
      generated, static_cast<std::uint64_t>(state_->trace.size()))));
  std::size_t index = state_->trace_head;
  for (std::size_t copied = 0u; copied < state_->trace.size(); ++copied) {
    ::rund::TraceRecord record = state_->trace[index];
    if (record.sequence >= begin.next) {
      record.sequence = record.sequence - begin.next + 1u;
      result.records.push_back(record);
    }
    ++index;
    if (index == state_->trace.size()) {
      index = 0u;
    }
  }
  const std::uint64_t retained =
      static_cast<std::uint64_t>(result.records.size());
  result.dropped = generated > retained ? generated - retained : 0u;
  return result;
}

::rund::Session::Result
Runtime::run_scope(const ::rund::replay::detail::scope::Plan &plan,
                   void *const callback, const ScopeCallback invoke,
                   const TraceCapture trace_capture,
                   ::rund::replay::detail::scope::Generation *const generation,
                   ::rund::replay::detail::scope::Lease *const lease,
                   ::rund::replay::detail::scope::Timing *const outer_timing) {
  const bool detail_enabled =
      state_->telemetry &&
      state_->telemetry.level() == ::rund::telemetry::Level::Detail;
  std::optional<::rund::replay::detail::scope::Timing> local_timing{};
  if (outer_timing == nullptr) {
    local_timing.emplace(detail_enabled);
  }
  ::rund::replay::detail::scope::Timing &timing =
      outer_timing == nullptr ? *local_timing : *outer_timing;
  const bool owns_timing = outer_timing == nullptr;
  const auto publish_timing = [&](::rund::Session::Result &result) {
    const ::rund::telemetry::Detail detail =
        owns_timing ? timing.stop() : timing.detail();
    const std::uint8_t reads = timing.reads();
    ::rund::detail::session::ResultAccess::timing(result, detail, reads);
  };
  const ScopeAdmission admission = enter_scope();
  if (!admission) {
    ::rund::Session::Result result =
        ::rund::detail::session::ResultAccess::fail(admission.code());
    publish_timing(result);
    return result;
  }

  struct ScopeExit final {
    Runtime *runtime = nullptr;
    ::rund::replay::detail::scope::Generation *generation = nullptr;
    ::rund::replay::detail::scope::Lease *lease = nullptr;
    std::uint64_t generation_value = 0u;
    ~ScopeExit() {
      if (generation != nullptr && generation_value != 0u) {
        std::uint64_t expected = generation_value;
        (void)generation->active.compare_exchange_strong(
            expected, 0u, std::memory_order_release, std::memory_order_relaxed);
      }
      if (lease != nullptr) {
        *lease = {};
      }
      if (runtime != nullptr) {
        runtime->leave_scope();
      }
    }
  } scope_exit{.runtime = this, .generation = generation, .lease = lease};

  if (generation != nullptr) {
    std::uint64_t value =
        generation->next.fetch_add(1u, std::memory_order_relaxed);
    if (value == 0u) {
      value = generation->next.fetch_add(1u, std::memory_order_relaxed);
    }
    generation->active.store(value, std::memory_order_release);
    scope_exit.generation_value = value;
    if (lease != nullptr) {
      *lease = ::rund::replay::detail::scope::Lease{
          .generation = &generation->active, .value = value};
    }
  }

  const TraceCursor trace_begin =
      trace_capture == TraceCapture::Scope ? begin_trace() : TraceCursor{};
  const auto result_trace = [&]() {
    return trace_capture == TraceCapture::Scope ? capture_trace(trace_begin)
                                                : ::rund::Trace{};
  };

  auto provider_scope = runtime_detail::InstallKernelScope(*this);
  auto task_scope = this->task_scope(admission.host, plan);
  if (!task_scope) {
    ::rund::Session::Result result =
        ::rund::detail::session::ResultAccess::fail(task_scope.code(),
                                                    result_trace());
    publish_timing(result);
    return result;
  }
  state_->active_scope.store(task_scope.id(), std::memory_order_release);
  timing.work();

  bool callback_failed = false;
  {
    runtime_detail::RuntimeActiveScope active(this);
    try {
      invoke(callback);
    } catch (...) {
      callback_failed = true;
    }
  }

  const task::Status task_drain = task_scope.drain();
  timing.finish();
  const ReasonCode result_code = callback_failed
                                     ? ReasonCode::RuntimeScopeCallbackFailed
                                 : !task_drain ? task_drain.code()
                                               : ReasonCode::Ok;
  ::rund::Session::Result result =
      capture_result(task_scope, result_code, result_trace());
  publish_timing(result);
  return result;
}

Runtime::TaskScopeFrame::TaskScopeFrame(
    std::shared_ptr<void> lifetime, void *const scheduler, std::mutex &control,
    const ::rund::replay::detail::scope::Plan &plan) noexcept
    : previous_(Scheduler::Active()), scheduler_(scheduler),
      lifetime_(std::move(lifetime)), control_(control) {
  if (scheduler_ == nullptr || lifetime_ == nullptr) {
    code_ = scheduler_ == nullptr
                ? ReasonCode::NodeRuntimeMissing
                : ReasonCode::TaskInvalid;
    return;
  }
  Scheduler::SetActive(static_cast<Scheduler *>(scheduler_));
  if (!static_cast<Scheduler *>(scheduler_)->InstallPlan(plan)) {
    Scheduler::SetActive(static_cast<Scheduler *>(previous_));
    code_ = ReasonCode::ReplayScopePrepareFailed;
    return;
  }
  plan_installed_ = true;
  const ScopeToken scope = static_cast<Scheduler *>(scheduler_)->BeginScope();
  if (scope.code != ReasonCode::Ok) {
    static_cast<Scheduler *>(scheduler_)->ClearPlan();
    plan_installed_ = false;
    Scheduler::SetActive(static_cast<Scheduler *>(previous_));
    code_ = scope.code;
    return;
  }
  scope_id_ = scope.scope_id;
  previous_scope_id_ = scope.previous_scope_id;
  observation_begin_ = scope.observation_begin;
  event_begin_ = scope.event_begin;
  installed_ = true;
  code_ = ReasonCode::Ok;
}

Runtime::TaskScopeFrame::~TaskScopeFrame() {
  if (installed_ && !drained_) {
    (void)drain();
  }
  if (plan_installed_ && scheduler_ != nullptr) {
    static_cast<Scheduler *>(scheduler_)->ClearPlan();
    plan_installed_ = false;
  }
  if (installed_) {
    Scheduler::SetActive(static_cast<Scheduler *>(previous_));
  }
  scheduler_ = nullptr;
}

task::Status Runtime::TaskScopeFrame::drain() noexcept {
  if (!installed_) {
    return task::Status::fail(scheduler_ == nullptr
                                  ? ReasonCode::NodeRuntimeMissing
                                  : ReasonCode::TaskInvalid);
  }
  if (drained_) {
    return task::Status::success();
  }
  const task::Status result =
      static_cast<Scheduler *>(scheduler_)
          ->EndScope(ScopeToken{.scope_id = scope_id_,
                                .previous_scope_id = previous_scope_id_,
                                .code = ReasonCode::Ok});
  drained_ = true;
  return result;
}

ScopeEvidence Runtime::TaskScopeFrame::evidence() {
  return scheduler_ != nullptr
             ? static_cast<Scheduler *>(scheduler_)
                   ->CaptureScope(observation_begin_, event_begin_)
             : ScopeEvidence{};
}

} // namespace rund::node
