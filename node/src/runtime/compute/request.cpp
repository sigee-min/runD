#include "request.hpp"
#include "../runtime/local.hpp"
#include "local.hpp"

#include <rund/task/coroutine.hpp>
#include <rund/task/handle/spawn.hpp>

#include <chrono>
#include <utility>

namespace rund::compute {
namespace {

using ComputeHost = node::runtime_detail::ComputeHostState;
using ComputeState = node::compute_detail::TaskState;

[[nodiscard]] std::shared_ptr<ComputeHost>
ComputeHostFrom(const std::shared_ptr<void> &host) noexcept {
  return std::static_pointer_cast<ComputeHost>(host);
}

[[nodiscard]] std::shared_ptr<ComputeHost>
ComputeHostFrom(const std::weak_ptr<void> &host) noexcept {
  return std::static_pointer_cast<ComputeHost>(host.lock());
}

[[nodiscard]] ComputeState *ComputeStateFrom(void *state) noexcept {
  return static_cast<ComputeState *>(state);
}

[[nodiscard]] Reason SubmissionAdmissionFailure(
    const node::runtime_detail::ComputeHostAdmission admission) noexcept {
  switch (admission) {
  case node::runtime_detail::ComputeHostAdmission::Open:
    return Reason::Ok;
  case node::runtime_detail::ComputeHostAdmission::Draining:
    return Reason::RuntimeDraining;
  case node::runtime_detail::ComputeHostAdmission::Standby:
    return Reason::RuntimeNotRunning;
  case node::runtime_detail::ComputeHostAdmission::Offline:
    return Reason::RuntimeMissing;
  }
}

void RetireCompute(const std::shared_ptr<void> &host, void *state) noexcept {
  ComputeState *typed = ComputeStateFrom(state);
  node::Retire(ComputeHostFrom(host), typed);
}

void ReleaseCompute(void *&state) noexcept {
  ComputeState *typed = ComputeStateFrom(state);
  node::Release(typed);
  state = typed;
}

} // namespace

Submission Request::submit() const noexcept {
  const node::compute_detail::Operation operation =
      node::compute_detail::make_operation(operation_, operations_);
  if (!operation) {
    return Submission{Status::fail(Reason::TaskInvalid)};
  }
  const Result<Backend> backend = node::OperationBackend(operation);
  if (!backend) {
    return Submission{Status::fail(backend.reason())};
  }
  const std::shared_ptr<ComputeHost> host = ComputeHostFrom(host_);
  if (host == nullptr) {
    return Submission{Status::fail(Reason::RuntimeMissing)};
  }
  ComputeState *task = nullptr;
  try {
    {
      std::lock_guard lock{host->mutex};
      const Reason rejected =
          SubmissionAdmissionFailure(host->lifecycle.admission());
      if (rejected != Reason::Ok) {
        return Submission{Status::fail(rejected)};
      }
      if (host->scope_active && !node::OwnsSchedulerControl(*host)) {
        return Submission{Status::fail(Reason::RuntimeBusy)};
      }
      if (*backend == Backend::Cpu &&
          node::OperationWorkers(operation) != host->workers) {
        return Submission{Status::fail(Reason::NodeHostWidthMismatch)};
      }
      task = node::ClaimTask(*host, operation);
      if (task == nullptr) {
        return Submission{Status::fail(Reason::TaskCapacity)};
      }
      ++host->outstanding;
      task->submitted = true;
    }

    const Status queued = node::ReserveOperation(task->operation);
    if (!queued) {
      {
        std::lock_guard lock{host->mutex};
        if (host->outstanding != 0u) {
          --host->outstanding;
        }
        task->submitted = false;
      }
      host->drained.notify_all();
      node::Release(task);
      return Submission{queued};
    }
    node::Signal(host.get(), TraceEvent::ComputeSubmitted);
    node::Signal(host.get(), TraceEvent::ComputeAdmitted);

    const bool scheduler_task = node::InSchedulerTask(*host);
    const auto spawn_coordinator = [&] {
      node::Scheduler *const prior = node::Scheduler::Active();
      node::Scheduler::SetActive(&host->scheduler);
      task::Task<void> coordinator = *backend != Backend::Cpu
                                         ? node::RunAccelCoordinator(task)
                                         : node::RunCpuCoordinator(task);
      void *const frame = ::rund::detail::task::CoroutineAddress(coordinator);
      const std::uint32_t frame_bytes =
          ::rund::detail::task::frame::Bytes(frame);
      const bool frame_reused = ::rund::detail::task::frame::Reused(frame);
      const ::rund::detail::task::CoroutineStart start =
          ::rund::detail::task::TakeCoroutine(std::move(coordinator));
      node::Scheduler::SetActive(prior);
      const ::rund::detail::task::Spawned spawned =
          host->scheduler.SpawnObserved("compute", start);
      task->handle = spawned.task;
      task->completion = spawned.result;
      if (task->handle) {
        task->frame_bytes = frame_bytes;
        node::RecordOperationFrame(
            task->operation, frame_bytes, frame_reused,
            node::scheduler_access::CoroutineFrameByteLimit(host->scheduler));
      }
      if (task->handle && !scheduler_task) {
        const bool dispatched = host->scheduler.DispatchExternal(task->handle);
        task->external_started.store(dispatched, std::memory_order_release);
      }
    };
    if (node::OwnsSchedulerControl(*host)) {
      spawn_coordinator();
    } else {
      std::lock_guard control{host->control};
      spawn_coordinator();
    }
    if (!task->handle) {
      const Status failure =
          Status::fail(node::SpawnReason(task->handle.code()));
      const Status status = node::FinishOperation(*task, failure);
      node::Complete(task, status, node::OperationEvidence(task->operation));
      node::Release(task);
      return Submission{failure};
    }
    return Submission{host, task};
  } catch (...) {
    if (task != nullptr) {
      const Status failure = Status::fail(Reason::TaskCapacity);
      const Status status = node::FinishOperation(*task, failure);
      node::Complete(task, status, node::OperationEvidence(task->operation));
      node::Release(task);
      return Submission{failure};
    }
    return Submission{Status::fail(Reason::TaskCapacity)};
  }
}

Submission::~Submission() {
  RetireCompute(host_, state_);
  ReleaseCompute(state_);
}

Submission::Submission(Submission &&other) noexcept
    : host_(std::move(other.host_)),
      state_(std::exchange(other.state_, nullptr)),
      immediate_(other.immediate_) {}

Submission &Submission::operator=(Submission &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  RetireCompute(host_, state_);
  ReleaseCompute(state_);
  host_ = std::move(other.host_);
  state_ = std::exchange(other.state_, nullptr);
  immediate_ = other.immediate_;
  return *this;
}

Poll Submission::poll() const noexcept {
  if (state_ == nullptr) {
    return Poll{false, false, true, immediate_.reason()};
  }
  ComputeState *const state = ComputeStateFrom(state_);
  std::lock_guard lock{state->mutex};
  const task::Poll completion =
      state->completion.poll == nullptr
          ? task::Poll{.phase = task::Phase::Failed,
                       .code = ::rund::ReasonCode::TaskHandleStale}
          : state->completion.poll(state->completion.authority,
                                   state->completion.slot,
                                   state->completion.generation);
  const bool completed = completion.terminal();
  return Poll{state->submitted,
              state->backend_submitted.load(std::memory_order_acquire),
              completed, completed ? state->status.reason() : Reason::Ok};
}

Poll Submission::wait_for(
    const std::chrono::nanoseconds timeout) const noexcept {
  if (state_ == nullptr || timeout <= std::chrono::nanoseconds::zero()) {
    return poll();
  }
  const std::shared_ptr<ComputeHost> host = ComputeHostFrom(host_);
  ComputeState *const state = ComputeStateFrom(state_);
  if (host == nullptr || state == nullptr) {
    return Poll{false, false, true, Reason::RuntimeMissing};
  }
  {
    std::unique_lock lock{host->mutex};
    host->drained.wait_for(lock, timeout, [state] {
      return state->terminal_phase.load(std::memory_order_acquire) ==
             node::compute_detail::TerminalPhase::Complete;
    });
  }
  return poll();
}

task::Handle Submission::handle() const noexcept {
  if (state_ == nullptr) {
    return {};
  }
  ComputeState *const state = ComputeStateFrom(state_);
  std::lock_guard lock{state->mutex};
  return state->handle;
}

Completion Submission::wait() const noexcept {
  if (state_ == nullptr) {
    return Completion{immediate_, {}};
  }
  RetireCompute(host_, state_);
  ComputeState *const state = ComputeStateFrom(state_);
  std::lock_guard lock{state->mutex};
  return Completion{state->status, state->stats};
}

Status Submission::cancel() const noexcept {
  if (state_ == nullptr) {
    return immediate_;
  }
  ComputeState *const state = ComputeStateFrom(state_);
  const node::compute_detail::CancelClaim claim =
      node::compute_detail::RequestCancel(state->terminal_phase);
  if (claim == node::compute_detail::CancelClaim::Accept ||
      claim == node::compute_detail::CancelClaim::Cancelled) {
    state->cancel_requested.store(true, std::memory_order_release);
    return Status::success();
  }
  if (claim == node::compute_detail::CancelClaim::Closed) {
    return Status::fail(Reason::AlreadyCompleted);
  }
  return Status::fail(Reason::CompletionInvalid);
}

} // namespace rund::compute
