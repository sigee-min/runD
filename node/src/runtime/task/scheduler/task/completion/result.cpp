#include "local.hpp"

#include <rund/task/coroutine.hpp>

namespace rund::node {

task::Status CompletionPool::prepare_raw(
    const CompletionLease lease, void *const value, const std::size_t bytes,
    const std::size_t alignment, const void *const type, const MoveFn move,
    const DestroyFn destroy) noexcept {
  auto *const store = static_cast<Store *>(lease.authority);
  if (store == nullptr || !store->contains(lease.slot)) {
    return task::Status::fail(ReasonCode::TaskHandleStale);
  }
  Store::Cell &cell = store->at(lease.slot);
  std::lock_guard lock{store->stripe(lease.slot).mutex};
  if (cell.generation != lease.generation || cell.phase == task::Phase::Idle) {
    return task::Status::fail(ReasonCode::TaskHandleStale);
  }
  if (cell.phase != task::Phase::Committing ||
      cell.outcome != CompletionOutcome::Pending) {
    return task::Status::fail(ReasonCode::TaskStateTransitionInvalid);
  }
  if (bytes > store->limits.result_bytes ||
      alignment > store->limits.result_alignment) {
    return task::Status::fail(ReasonCode::TaskResultTooLarge);
  }
  if (cell.result == nullptr) {
    const std::size_t block_bytes =
        static_cast<std::size_t>(store->limits.result_alignment) +
        store->limits.result_bytes;
    cell.result = ::operator new(
        block_bytes, std::align_val_t{store->limits.result_alignment},
        std::nothrow);
    if (cell.result == nullptr) {
      return task::Status::fail(ReasonCode::TaskCompletionCapacity);
    }
  }
  try {
    move(store->result(lease.slot), value);
  } catch (...) {
    return task::Status::fail(ReasonCode::TaskFailed);
  }
  store->result_header(cell) =
      Store::ResultHeader{.type = type, .destroy = destroy};
  cell.outcome = CompletionOutcome::Value;
  cell.code = ReasonCode::Ok;
  return task::Status::success();
}

task::Status
CompletionPool::prepare_complete(const CompletionLease lease) noexcept {
  auto *const store = static_cast<Store *>(lease.authority);
  if (store == nullptr || !store->contains(lease.slot)) {
    return task::Status::fail(ReasonCode::TaskHandleStale);
  }
  Store::Cell &cell = store->at(lease.slot);
  std::lock_guard lock{store->stripe(lease.slot).mutex};
  if (cell.generation != lease.generation || cell.phase == task::Phase::Idle) {
    return task::Status::fail(ReasonCode::TaskHandleStale);
  }
  if (cell.phase != task::Phase::Committing ||
      cell.outcome != CompletionOutcome::Pending) {
    return task::Status::fail(ReasonCode::TaskStateTransitionInvalid);
  }
  cell.code = ReasonCode::Ok;
  cell.outcome = CompletionOutcome::Empty;
  return task::Status::success();
}

task::Status CompletionPool::complete(const CompletionLease lease) noexcept {
  const task::Status prepared = prepare_complete(lease);
  return prepared ? publish(lease) : prepared;
}

task::Status CompletionPool::prepare(
    const CompletionLease lease,
    const ::rund::detail::task::CoroutineResult &result) noexcept {
  if (result.code != ReasonCode::Ok) {
    const task::Status failed = prepare_failure(lease, result.code);
    return failed ? task::Status::fail(result.code) : failed;
  }
  if (!result.has_value) {
    return prepare_complete(lease);
  }
  if (result.value == nullptr || result.type == nullptr ||
      result.move == nullptr || result.destroy == nullptr) {
    return task::Status::fail(ReasonCode::TaskFailed);
  }
  return prepare_raw(lease, result.value, result.bytes, result.alignment,
                     result.type, result.move, result.destroy);
}

task::Status CompletionPool::prepare_failure(const CompletionLease lease,
                                             const ReasonCode code) noexcept {
  auto *const store = static_cast<Store *>(lease.authority);
  if (store == nullptr || !store->contains(lease.slot)) {
    return task::Status::fail(ReasonCode::TaskHandleStale);
  }
  Store::Cell &cell = store->at(lease.slot);
  std::lock_guard lock{store->stripe(lease.slot).mutex};
  if (cell.generation != lease.generation || cell.phase == task::Phase::Idle) {
    return task::Status::fail(ReasonCode::TaskHandleStale);
  }
  if (CompletionTerminal(cell.phase)) {
    return task::Status::fail(ReasonCode::TaskStateTransitionInvalid);
  }
  store->destroy_value(lease.slot, cell);
  cell.phase = task::Phase::Committing;
  cell.code = CompletionFailure(code);
  cell.outcome = CompletionOutcome::Failure;
  return task::Status::success();
}

task::Status CompletionPool::copy_raw(const CompletionLease lease,
                                      void *const out, const void *const type,
                                      const CopyFn copy) {
  auto *const store = static_cast<Store *>(lease.authority);
  if (store == nullptr || !store->contains(lease.slot)) {
    return task::Status::fail(ReasonCode::TaskHandleStale);
  }
  Store::Cell &cell = store->at(lease.slot);
  std::lock_guard lock{store->stripe(lease.slot).mutex};
  if (cell.generation != lease.generation || cell.phase == task::Phase::Idle) {
    return task::Status::fail(ReasonCode::TaskHandleStale);
  }
  if (cell.phase != task::Phase::Completed ||
      cell.outcome != CompletionOutcome::Value) {
    return task::Status::fail(
        cell.code == ReasonCode::Ok ? ReasonCode::TaskFailed : cell.code);
  }
  if (store->result_header(cell).type != type) {
    return task::Status::fail(ReasonCode::TaskResultTypeMismatch);
  }
  try {
    copy(out, store->result(lease.slot));
  } catch (...) {
    return task::Status::fail(ReasonCode::TaskFailed);
  }
  return task::Status::success();
}

} // namespace rund::node
