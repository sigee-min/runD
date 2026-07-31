#include "../../../host/net/operation.hpp"
#include "../../../host/net/ready/ticket.hpp"
#include "../../../host/net/scheduler.hpp"
#include "../../../host/net/socket/access.hpp"
#include "../../platform/io.hpp"
#include "../scheduler/access.hpp"
#include "../scheduler/state/model/task.hpp"
#include "../scheduler/state/storage.hpp"

#include <rund/net/ready/set.hpp>
#include <rund/net/ready/timed.hpp>
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

namespace rund::detail::task {

AwaitDecision AwaitAccess::SuspendCoroutineYield() noexcept {
  ::rund::node::Scheduler *const scheduler = ::rund::node::Scheduler::Active();
  if (scheduler == nullptr) {
    return ::rund::node::FailYield(ReasonCode::NodeRuntimeMissing);
  }
  return scheduler->Yield();
}

AwaitDecision AwaitAccess::SuspendCoroutineSleep(
    const std::chrono::nanoseconds duration) noexcept {
  ::rund::node::Scheduler *const scheduler = ::rund::node::Scheduler::Active();
  if (scheduler == nullptr) {
    return ::rund::node::FailSleep(ReasonCode::NodeRuntimeMissing);
  }
  return scheduler->Sleep(duration);
}

AwaitDecision AwaitAccess::BeginCoroutineJoinAwait(
    const ::rund::task::Handle &handle) noexcept {
  ::rund::node::Scheduler *const scheduler = ::rund::node::Scheduler::Active();
  if (scheduler == nullptr) {
    return AwaitDecision{
        .status = ::rund::task::Status::fail(ReasonCode::NodeRuntimeMissing)};
  }
  return scheduler->BeginJoinAwait(&handle, 1u);
}

::rund::task::Status
AwaitAccess::CompleteCoroutineJoinAwait(const AwaitDecision decision) noexcept {
  if (!decision.suspend) {
    return decision.status;
  }
  ::rund::node::Scheduler *const scheduler = ::rund::node::Scheduler::Active();
  if (scheduler == nullptr) {
    return ::rund::task::Status::fail(ReasonCode::NodeRuntimeMissing);
  }
  scheduler->EnsureCurrentCommit();
  ::rund::node::TaskRecord *const record =
      scheduler->state_->Find(scheduler->CurrentTaskId());
  if (record == nullptr || !record->coroutine_task) {
    return ::rund::task::Status::fail(ReasonCode::TaskContextMissing);
  }
  const ReasonCode code = record->wait_result;
  record->wait_result = ReasonCode::Ok;
  record->wait_id = 0u;
  record->wait_source_id = 0u;
  record->wait_token = 0u;
  return code == ReasonCode::Ok ? ::rund::task::Status::success()
                                : ::rund::task::Status::fail(code);
}

void AwaitAccess::RetireCoroutineTask(
    const ::rund::task::Handle &handle) noexcept {
  ::rund::node::Scheduler *const scheduler = ::rund::node::Scheduler::Active();
  if (scheduler == nullptr || !handle) {
    return;
  }
  scheduler->EnsureCurrentCommit();
  const std::uint64_t task_id = handle.id();
  ::rund::node::TaskRecord *const record =
      scheduler->state_->FindAt(scheduler->state_->IndexFor(task_id), task_id);
  if (!scheduler->Matches(handle, record) || record == nullptr ||
      (record->state != ::rund::node::TaskState::Completed &&
       record->state != ::rund::node::TaskState::Failed)) {
    return;
  }
  scheduler->RetireTask(*record);
}

::rund::task::IoResult
AwaitAccess::CompleteCoroutineIoAwait(const IoDecision decision) noexcept {
  if (!decision.suspend) {
    return decision.status ? ::rund::task::IoResult::success(decision.revents)
                           : ::rund::task::IoResult::fail(
                                 decision.status.code(), decision.revents);
  }
  ::rund::node::Scheduler *const scheduler = ::rund::node::Scheduler::Active();
  if (scheduler == nullptr) {
    return ::rund::task::IoResult::fail(ReasonCode::NodeRuntimeMissing);
  }
  scheduler->EnsureCurrentCommit();
  ::rund::node::TaskRecord *const record =
      scheduler->state_->Find(scheduler->CurrentTaskId());
  if (record == nullptr || !record->coroutine_task) {
    return ::rund::task::IoResult::fail(ReasonCode::TaskContextMissing);
  }
  const ReasonCode code = record->io_result;
  const short revents = record->io_revents;
  record->io_result = ReasonCode::Ok;
  record->io_revents = 0;
  return code == ReasonCode::Ok ? ::rund::task::IoResult::success(revents)
                                : ::rund::task::IoResult::fail(code, revents);
}

IoDecision
AwaitAccess::SuspendCoroutineIo(const int fd, const short interest,
                                const std::uint64_t host_handle_id,
                                const std::uint64_t fd_generation) noexcept {
  ::rund::node::Scheduler *const scheduler = ::rund::node::Scheduler::Active();
  if (scheduler == nullptr) {
    return ::rund::node::FailIo(ReasonCode::NodeRuntimeMissing);
  }
  return scheduler->WaitReactor(fd, interest, host_handle_id, fd_generation);
}

IoDecision
AwaitAccess::SuspendCoroutineNetIo(const ::rund::net::SocketView socket,
                                   const short interest) noexcept {
  return ::rund::net::WaitReactor(socket, interest);
}

::rund::task::IoResult
AwaitAccess::CompleteCoroutineNetIo(const IoDecision decision) noexcept {
  return CompleteCoroutineIoAwait(decision);
}

::rund::net::ready::Ticket AwaitAccess::CompleteCoroutineTimedReadyAwait(
    ::rund::net::ready::timed::Wait operation) noexcept {
  if (!operation.suspended_) {
    return std::move(operation.result_);
  }
  ::rund::node::Scheduler *const scheduler = ::rund::node::Scheduler::Active();
  if (scheduler == nullptr) {
    return ::rund::net::ready::detail::Access::make(
        ReasonCode::NodeRuntimeMissing, operation.socket_,
        operation.public_interest_);
  }
  const std::uint64_t current_task_id = scheduler->CurrentTaskId();
  if (operation.task_id_ != 0u && operation.task_id_ != current_task_id) {
    return ::rund::net::ready::detail::Access::make(
        ReasonCode::TaskContextMissing, operation.socket_,
        operation.public_interest_);
  }
  scheduler->EnsureCurrentCommit();
  ::rund::node::TaskRecord *const record =
      scheduler->state_->Find(current_task_id);
  if (record == nullptr || !record->coroutine_task) {
    return ::rund::net::ready::detail::Access::make(
        ReasonCode::TaskContextMissing, operation.socket_,
        operation.public_interest_);
  }
  const ReasonCode code = record->io_result;
  const short revents = record->io_revents;
  record->wait_id = 0u;
  record->io_result = ReasonCode::Ok;
  record->io_revents = 0;
  if (code == ReasonCode::IoTimedOut) {
    return ::rund::net::ready::detail::Access::make(
        ReasonCode::IoTimedOut, operation.socket_, operation.public_interest_,
        revents);
  }
  if (code != ReasonCode::Ok) {
    return ::rund::net::ready::detail::Access::make(
        code, operation.socket_, operation.public_interest_, revents);
  }
  return ::rund::net::ready::detail::Access::make(
      ReasonCode::Ok, operation.socket_, operation.public_interest_, revents);
}

::rund::net::ready::timed::Wait AwaitAccess::SuspendCoroutineTimedReady(
    ::rund::net::ready::timed::Wait operation) noexcept {
  ::rund::node::Scheduler *const scheduler = ::rund::node::Scheduler::Active();
  if (scheduler == nullptr) {
    operation.result_ = ::rund::net::ready::detail::Access::make(
        ReasonCode::NodeRuntimeMissing, operation.socket_,
        operation.public_interest_);
    operation.deferred_ = false;
    operation.suspended_ = false;
    return operation;
  }
  const IoDecision io = ::rund::net::WaitReactorTimed(
      *scheduler, operation.socket_, operation.interest_,
      std::chrono::nanoseconds{operation.timeout_ns_},
      operation.stop_scheduler_id_, operation.stop_source_id_,
      operation.stop_generation_, operation.stop_epoch_);
  operation.deferred_ = false;
  operation.suspended_ = io.suspend;
  operation.task_id_ = io.suspend ? scheduler->CurrentTaskId() : 0u;
  operation.result_ = ::rund::net::ready::detail::Access::make(
      io.status.code(), operation.socket_, operation.public_interest_,
      io.revents);
  return operation;
}

::rund::net::ready::many::Wait AwaitAccess::SuspendCoroutineReadyMany(
    ::rund::net::ready::many::Wait operation) noexcept {
  const ::rund::net::ready::many::detail::Context context =
      ::rund::net::ready::many::detail::Access::Snapshot(operation);
  ::rund::node::Scheduler *const scheduler = ::rund::node::Scheduler::Active();
  if (scheduler == nullptr) {
    ::rund::net::ready::many::detail::Access::Reject(
        operation, ReasonCode::NodeRuntimeMissing);
    return operation;
  }
  const std::optional<std::chrono::nanoseconds> timeout =
      context.has_timeout
          ? std::optional<std::chrono::nanoseconds>{std::chrono::nanoseconds{
                context.timeout_ns}}
          : std::nullopt;
  ::rund::net::ready::many::Wait suspended =
      context.ready_set_id == 0u
          ? scheduler->WaitReactorMany(
                context.requests, context.out, timeout, context.budget,
                context.stop_scheduler_id, context.stop_source_id,
                context.stop_generation, context.stop_epoch)
          : scheduler->WaitReadySet(
                ::rund::net::ready::Set{
                    .id = context.ready_set_id,
                    .generation = context.ready_set_generation,
                },
                context.out, timeout, context.budget);
  ::rund::net::ready::many::detail::Access::Restore(suspended, context);
  return suspended;
}

::rund::net::ready::many::Result AwaitAccess::CompleteCoroutineReadyMany(
    ::rund::net::ready::many::Wait operation) noexcept {
  using ManyAccess = ::rund::net::ready::many::detail::Access;
  if (!ManyAccess::Suspended(operation)) {
    return ManyAccess::ResultOf(operation);
  }
  ::rund::node::Scheduler *const scheduler = ::rund::node::Scheduler::Active();
  if (scheduler == nullptr) {
    return ::rund::net::ready::many::Result{ReasonCode::NodeRuntimeMissing};
  }
  ::rund::net::ready::many::Wait resumed = scheduler->ResumeReactorMany(
      ManyAccess::Output(operation), ManyAccess::Group(operation));
  return ManyAccess::Suspended(resumed)
             ? ::rund::net::ready::many::Result{ReasonCode::TaskContextMissing}
             : ManyAccess::ResultOf(resumed);
}
} // namespace rund::detail::task

namespace rund::net::ready::timed {

bool Awaiter::await_suspend(std::coroutine_handle<>) noexcept {
  if (operation_.deferred_) {
    operation_ = ::rund::detail::task::AwaitAccess::SuspendCoroutineTimedReady(
        std::move(operation_));
  }
  return operation_.suspended_;
}

Ticket Awaiter::await_resume() noexcept {
  return ::rund::detail::task::AwaitAccess::CompleteCoroutineTimedReadyAwait(
      std::move(operation_));
}

} // namespace rund::net::ready::timed
