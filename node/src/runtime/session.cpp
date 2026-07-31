#include <rund/session.hpp>

#include <kernel/program/executor/model.hpp>
#include <node/runtime/replay/hash.hpp>
#include <node/runtime/runtime.hpp>

#include "replay/scope/session.hpp"
#include "runtime/local.hpp"
#include "session/result.hpp"
#include "session/state.hpp"
#include "session/status.hpp"

#include <algorithm>
#include <type_traits>

namespace rund {
namespace {

template <class T>
concept HasDirectDrain = requires(T &value) { value.drain(); };

template <class T>
concept HasDirectClose = requires(T &value) { value.close(); };

static_assert(!HasDirectDrain<node::Runtime>);
static_assert(!HasDirectClose<node::Runtime>);
static_assert(!std::is_default_constructible_v<node::Runtime>);

[[nodiscard]] std::uint32_t workers(const std::uint32_t requested) noexcept {
  if (requested != 0u) {
    return requested;
  }
  const std::uint32_t hint = kernel::executor_detail::HostParallelWorkerHint();
  return hint == 0u ? 1u : hint;
}

[[nodiscard]] Session::Status unopened(const ::rund::ReasonCode code) noexcept {
  return ::rund::detail::session::StatusAccess::make(
      code, ::rund::SessionState::Unconfigured);
}

[[nodiscard]] ::rund::ReasonCode
PrepareReplayStorage(::rund::replay::Storage &storage) noexcept {
  if (storage.mode != ::rund::replay::StorageMode::Spill) {
    return ::rund::ReasonCode::Ok;
  }
  if (storage.max_allocated_bytes == 0u) {
    return ::rund::ReasonCode::HostReplayStorageInvalid;
  }

  if (!storage.budget) {
    storage.budget = ::rund::storage::Budget{storage.max_allocated_bytes};
    return storage.budget ? ::rund::ReasonCode::Ok : storage.budget.code();
  }

  const ::rund::storage::Report shared = storage.budget.report();
  if (!shared) {
    return shared.code;
  }
  const std::uint64_t session_capacity =
      std::min(storage.max_allocated_bytes, shared.capacity_bytes);
  storage.budget = storage.budget.child(session_capacity);
  return storage.budget ? ::rund::ReasonCode::Ok : storage.budget.code();
}

} // namespace

Session::Session() : state_(std::make_unique<State>()) {}

Session::~Session() = default;

std::uint64_t Session::Result::trace_hash() const noexcept {
  return node::replay_detail::HashTrace(trace_);
}

Session::Status Session::open(SessionConfig config) {
  if (state_->runtime != nullptr) {
    return state_->runtime->configure(std::move(config));
  }

  try {
    config.workers = workers(config.workers);
    const ::rund::ReasonCode storage_code =
        PrepareReplayStorage(config.replay.storage);
    if (storage_code != ::rund::ReasonCode::Ok) {
      return unopened(storage_code);
    }
    ::rund::replay::Storage replay_storage = config.replay.storage;
    const ::rund::replay::Diagnostic replay_diagnostic =
        config.replay.diagnostic;
    const std::uint32_t replay_inputs = config.replay.input_capacity;
    const std::uint64_t replay_capacity = std::min(
        config.scheduler.host_payload_capacity_bytes, replay_storage.max_bytes);
    auto candidate = std::unique_ptr<node::Runtime>{new node::Runtime()};
    const Status configured = candidate->configure(std::move(config));
    if (!configured) {
      return configured;
    }
    const Status started = candidate->start();
    if (!started) {
      return started;
    }
    static_assert(std::is_nothrow_move_assignable_v<replay::Storage>);
    state_->replay_storage = std::move(replay_storage);
    state_->replay_diagnostic = replay_diagnostic;
    state_->replay_capacity = replay_capacity;
    state_->replay_inputs = replay_inputs;
    state_->runtime = std::move(candidate);
    return started;
  } catch (...) {
    return unopened(::rund::ReasonCode::TaskSchedulerAllocationFailed);
  }
}

Session::Status Session::drain() {
  return state_->runtime == nullptr
             ? unopened(::rund::ReasonCode::NotConfigured)
             : state_->runtime->shutdown(node::Runtime::Shutdown::Admission);
}

Session::Status Session::close() {
  return state_->runtime == nullptr
             ? unopened(::rund::ReasonCode::NotConfigured)
             : state_->runtime->shutdown(node::Runtime::Shutdown::Terminal);
}

Session::Result Session::run_scope(void *const callback,
                                   const ScopeCallback invoke) {
  if (state_->runtime == nullptr) {
    return ::rund::detail::session::ResultAccess::fail(
        ::rund::ReasonCode::RuntimeScopeNotStarted);
  }
  return state_->runtime->run_scope(
      ::rund::replay::detail::scope::Plan{}, callback, invoke,
      node::Runtime::TraceCapture::Scope, nullptr, nullptr, nullptr);
}

Session::Result
Session::run_scope(const ::rund::replay::detail::scope::Plan &plan,
                   void *const callback, const ScopeCallback invoke,
                   ::rund::replay::detail::scope::Lease *const lease,
                   ::rund::replay::detail::scope::Timing *const timing) {
  if (state_->runtime == nullptr) {
    return ::rund::detail::session::ResultAccess::fail(
        ::rund::ReasonCode::RuntimeScopeNotStarted);
  }
  return state_->runtime->run_scope(plan, callback, invoke,
                                    node::Runtime::TraceCapture::Scope,
                                    &state_->generation, lease, timing);
}

Session::Result Session::terminal(void *const callback,
                                  const ScopeCallback invoke) {
  if (state_->runtime == nullptr) {
    return ::rund::detail::session::ResultAccess::fail(
        ::rund::ReasonCode::RuntimeScopeNotStarted);
  }
  return state_->runtime->run_scope(
      ::rund::replay::detail::scope::Plan{}, callback, invoke,
      node::Runtime::TraceCapture::Deferred, nullptr, nullptr, nullptr);
}

Session::Snapshot Session::snapshot() const {
  return state_->runtime == nullptr ? Snapshot{} : state_->runtime->snapshot();
}

::rund::Trace Session::trace() const {
  return state_->runtime == nullptr ? ::rund::Trace{}
                                    : state_->runtime->trace();
}

::rund::Trace Session::take_trace() {
  return state_->runtime == nullptr
             ? ::rund::Trace{}
             : node::runtime_detail::RuntimeAccess::take_trace(
                   *state_->runtime);
}

Resources Session::resources() const {
  return state_->runtime == nullptr ? Resources{}
                                    : state_->runtime->resources();
}

void Session::emit(telemetry::Event &&event) noexcept {
  if (state_->runtime != nullptr) {
    state_->runtime->emit(std::move(event));
  }
}

} // namespace rund

namespace rund::replay::detail::scope {
namespace {

struct Call final {
  ::rund::Session *session = nullptr;
  void *callback = nullptr;
  Access::Callback invoke = nullptr;
  Lease lease{};
};

void Invoke(void *const raw) {
  auto &call = *static_cast<Call *>(raw);
  call.invoke(call.callback, *call.session, call.lease);
}

} // namespace

::rund::Session::Result Access::run(::rund::Session &session, const Plan &plan,
                                    void *const callback, const Callback invoke,
                                    Timing &timing) {
  Call call{.session = &session, .callback = callback, .invoke = invoke};
  return session.run_scope(plan, &call, Invoke, &call.lease, &timing);
}

bool Access::detail(::rund::Session &session) noexcept {
  if (session.state_->runtime == nullptr) {
    return false;
  }
  const auto &state = ::rund::node::runtime_detail::RuntimeAccess::state(
      *session.state_->runtime);
  return state.telemetry &&
         state.telemetry.level() == ::rund::telemetry::Level::Detail;
}

Prepared Access::prepare(
    ::rund::Session &session, std::vector<::rund::host::Event> events,
    ::rund::node::replay_detail::payload::Archive payloads) noexcept {
  if (session.state_->runtime == nullptr) {
    return Prepared{.code = ::rund::replay::Code::NotConfigured};
  }
  try {
    ::rund::node::replay_detail::payload::BuildResult built =
        ::rund::node::replay_detail::payload::Build(
            std::move(payloads), session.state_->replay_storage,
            session.state_->replay_diagnostic);
    if (!built.ok()) {
      return Prepared{.code = built.code};
    }
    return Prepared{
        .owner = std::make_shared<const Expected>(std::move(events),
                                                  std::move(built.store)),
        .code = ::rund::replay::Code::Ok,
    };
  } catch (...) {
    return Prepared{.code = ::rund::replay::Code::ScopePrepareFailed};
  }
}

std::uint64_t Access::capacity(::rund::Session &session) noexcept {
  return session.state_->runtime == nullptr ? 0u
                                            : session.state_->replay_capacity;
}

std::uint32_t Access::inputs(::rund::Session &session) noexcept {
  return session.state_->runtime == nullptr ? 0u
                                            : session.state_->replay_inputs;
}

} // namespace rund::replay::detail::scope
