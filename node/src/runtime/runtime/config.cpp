#include "../compute/state.hpp"
#include "../compute/workers.hpp"
#include "local.hpp"

#include <rund/host/random/seed.hpp>

#include "../../telemetry/compute.hpp"
#include "../resource/discovery.hpp"
#include "config/scheduler.hpp"

#include <utility>
#include <vector>

namespace rund::node {

using runtime_detail::LifecycleFail;
using runtime_detail::LifecyclePass;
using runtime_detail::ObserveLifecycle;
using runtime_detail::Runnable;
using runtime_detail::RuntimeActive;
using runtime_detail::RuntimeActiveScope;

::rund::Session::Status Runtime::configure(::rund::SessionConfig options) {
  if (RuntimeActive(this)) {
    return LifecycleFail(ReasonCode::RuntimeReentryForbidden,
                         ObserveLifecycle(*this));
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  RuntimeActiveScope active(this);

  if (options.compile.workers == 0u || options.compile.capacity == 0u) {
    return LifecycleFail(ReasonCode::RuntimeResourcesInvalid,
                         state_->lifecycle);
  }

  if (state_->lifecycle != ::rund::SessionState::Unconfigured) {
    return LifecycleFail(ReasonCode::AlreadyConfigured, state_->lifecycle);
  }
  if (!options.id) {
    return LifecycleFail(ReasonCode::RuntimeIdRequired, state_->lifecycle);
  }
  if (options.telemetry.level() != ::rund::telemetry::Level::Basic &&
      options.telemetry.level() != ::rund::telemetry::Level::Detail) {
    return LifecycleFail(ReasonCode::TelemetryLevelInvalid, state_->lifecycle);
  }
  if (!runtime_detail::HostReplayStorageValid(options.replay.storage)) {
    return LifecycleFail(ReasonCode::HostReplayStorageInvalid,
                         state_->lifecycle);
  }
  if (!runtime_detail::HostReplayDiagnosticValid(options.replay.diagnostic)) {
    return LifecycleFail(ReasonCode::HostReplayStorageInvalid,
                         state_->lifecycle);
  }
  runtime_detail::resource::Request resource_request{};
  resource_request.workers = options.workers;
  resource_request.require_verified_numa = options.require_verified_numa;
  resource_request.require_verified_affinity =
      options.require_verified_affinity;
  resource_request.require_verified_capacity =
      options.require_verified_worker_capacity;

  BackendSelection selection{};
  runtime_detail::resource::Result discovery{};
  ResourceEnvelope resources{};
  ::rund::SchedulerConfig scheduler{};
  std::shared_ptr<runtime_detail::ComputeHostState> compute_host{};
  std::vector<::rund::TraceRecord> trace{};
  try {
    scheduler = runtime_detail::NormalizeScheduler(
        options.scheduler, options.workers, options.trace_capacity);
    compute_host = std::make_shared<runtime_detail::ComputeHostState>();
    compute_host->workers = options.workers;
    compute_host->task_capacity = scheduler.task_capacity;
    compute_host->compile_resources = options.compile;
    compute_host->worker_capacity.assign(compute_host->workers, 1000u);
    compute_host->task_slots.configure(compute_host->task_capacity);
    compute_host->worker_batch_slots.configure(compute_host->task_capacity);

    const kernel::WorkerBackend backend =
        runtime_detail::MakeComputeSyncWorkerBackend(*compute_host);
    selection = BackendSelection{
        .code = ReasonCode::Ok,
        .requested_worker_width = options.workers,
        .backend = backend,
        .verified_numa = true,
    };
    compute_host->worker_backend = selection.backend;

    discovery = runtime_detail::resource::Resolve(std::move(selection),
                                                  resource_request);
    if (!discovery) {
      return LifecycleFail(discovery.code, state_->lifecycle);
    }
    resources = std::move(discovery.resources);

    trace.reserve(options.trace_capacity);
    constexpr std::uint32_t kWarmComputeTasks = 2u;
    const std::uint32_t warm_tasks =
        std::min(compute_host->task_capacity, kWarmComputeTasks);
    compute_host->tasks.reserve(warm_tasks);
    for (std::uint32_t index = 0u; index < warm_tasks; ++index) {
      auto task = std::make_unique<compute_detail::TaskState>();
      task->slot = index;
      compute_host->tasks.push_back(std::move(task));
      (void)compute_host->task_slots.release(index);
    }
    compute_host->async_worker_backend =
        runtime_detail::MakeComputeWorkerBackend(*compute_host);
    compute_host->signal_context = state_.get();
    compute_host->signal = [](void *const raw, const ::rund::TraceEvent event,
                              const ::rund::TraceCode code) noexcept {
      auto *const state = static_cast<State *>(raw);
      if (state == nullptr) {
        return;
      }
      std::lock_guard lock{state->mutex};
      state->AddTrace(event, code);
    };
    if (options.telemetry) {
      compute_host->emit_context = this;
      compute_host->emit = [](void *const raw, const compute::Status &status,
                              const compute::Stats &stats) noexcept {
        auto *const runtime = static_cast<Runtime *>(raw);
        if (runtime == nullptr) {
          return;
        }
        ::rund::telemetry::Event event =
            ::rund::telemetry::detail::ComputeEvent(
                stats, status.code(), ::rund::telemetry::Level::Detail);
        runtime->emit(std::move(event));
      };
    }
    if (!compute_host->scheduler.Configure(
            std::move(scheduler),
            rund::kernel::ParallelRuntimeProvider{
                .context = this,
                .acquire = runtime_detail::AcquireParallelRuntime,
            },
            std::move(options.replay),
            ::rund::host::random::RunSeed{.value = options.random_seed})) {
      return LifecycleFail(compute_host->scheduler.Code(), state_->lifecycle);
    }
  } catch (...) {
    return LifecycleFail(ReasonCode::TaskSchedulerAllocationFailed,
                         state_->lifecycle);
  }
  if (!compute_host->lifecycle.configure()) {
    return LifecycleFail(ReasonCode::NotRunnable, state_->lifecycle);
  }

  state_->id = options.id;
  state_->resources = std::move(resources);
  state_->telemetry = options.telemetry;
  state_->trace_capacity = options.trace_capacity;
  state_->trace = std::move(trace);
  state_->trace_head = 0u;
  state_->dropped_trace = 0u;
  state_->next_trace = 1u;
  state_->compute_host = std::move(compute_host);
  state_->lifecycle = ::rund::SessionState::Configured;
  state_->AddTrace(::rund::TraceEvent::RuntimeConfigured,
                   ::rund::TraceCode::runtime(ReasonCode::Ok));
  return LifecyclePass(state_->lifecycle);
}

Runtime::ScopeAdmission Runtime::enter_scope() {
  if (RuntimeActive(this)) {
    return ScopeAdmission{
        .status = LifecycleFail(ReasonCode::RuntimeReentryForbidden,
                                ObserveLifecycle(*this))};
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (!Runnable(state_->lifecycle) ||
      !static_cast<bool>(state_->resources.worker_backend) ||
      state_->compute_host == nullptr) {
    return ScopeAdmission{
        .status = LifecycleFail(ReasonCode::RuntimeScopeNotStarted,
                                state_->lifecycle)};
  }
  if (state_->scope_active) {
    return ScopeAdmission{
        .status =
            LifecycleFail(ReasonCode::RuntimeScopeBusy, state_->lifecycle)};
  }
  const std::shared_ptr<runtime_detail::ComputeHostState> host =
      state_->compute_host;
  {
    std::lock_guard host_lock{host->mutex};
    if (host->outstanding != 0u || host->scope_active) {
      return ScopeAdmission{
          .status =
              LifecycleFail(ReasonCode::RuntimeScopeBusy, state_->lifecycle)};
    }
    host->scope_active = true;
  }
  state_->scope_active = true;
  state_->active_scope.store(0u, std::memory_order_release);
  return ScopeAdmission{.status = LifecyclePass(state_->lifecycle),
                        .host = host};
}

void Runtime::leave_scope() noexcept {
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->compute_host != nullptr) {
      std::lock_guard host_lock{state_->compute_host->mutex};
      state_->compute_host->scope_active = false;
    }
    state_->scope_active = false;
    state_->active_scope.store(0u, std::memory_order_release);
  }
  state_->scope_drained.notify_all();
}

} // namespace rund::node
