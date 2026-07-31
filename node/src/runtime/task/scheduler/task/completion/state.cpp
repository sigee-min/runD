#include "local.hpp"

namespace rund::node {

task::Status CompletionPool::transition(const CompletionLease lease,
                                        const task::Phase next) noexcept {
  auto *const store = static_cast<Store *>(lease.authority);
  if (store == nullptr || !store->contains(lease.slot)) {
    return task::Status::fail(ReasonCode::TaskHandleStale);
  }
  Store::Cell &cell = store->at(lease.slot);
  std::lock_guard lock{store->stripe(lease.slot).mutex};
  if (cell.generation != lease.generation || cell.phase == task::Phase::Idle) {
    return task::Status::fail(ReasonCode::TaskHandleStale);
  }
  if (!CompletionAllowed(cell.phase, next)) {
    return task::Status::fail(ReasonCode::TaskStateTransitionInvalid);
  }
  cell.phase = next;
  return task::Status::success();
}

task::Poll CompletionPool::poll(const CompletionLease lease) noexcept {
  auto *const store = static_cast<Store *>(lease.authority);
  if (store == nullptr || !store->contains(lease.slot)) {
    return task::Poll{.phase = task::Phase::Failed,
                      .code = ReasonCode::TaskHandleStale};
  }
  Store::Cell &cell = store->at(lease.slot);
  std::lock_guard lock{store->stripe(lease.slot).mutex};
  if (cell.generation != lease.generation || cell.phase == task::Phase::Idle) {
    return task::Poll{.phase = task::Phase::Failed,
                      .code = ReasonCode::TaskHandleStale};
  }
  return task::Poll{.phase = cell.phase, .code = cell.code};
}

task::Status CompletionPool::publish(const CompletionLease lease) noexcept {
  auto *const store = static_cast<Store *>(lease.authority);
  if (store == nullptr || !store->contains(lease.slot)) {
    return task::Status::fail(ReasonCode::TaskHandleStale);
  }
  Store::Cell &cell = store->at(lease.slot);
  Store::Stripe &stripe = store->stripe(lease.slot);
  CompletionWaiter *waiters = nullptr;
  {
    std::lock_guard lock{stripe.mutex};
    if (cell.generation != lease.generation ||
        cell.phase == task::Phase::Idle) {
      return task::Status::fail(ReasonCode::TaskHandleStale);
    }
    if (cell.phase != task::Phase::Committing ||
        cell.outcome == CompletionOutcome::Pending) {
      return task::Status::fail(ReasonCode::TaskStateTransitionInvalid);
    }
    cell.phase = cell.outcome == CompletionOutcome::Failure
                     ? CompletionFailurePhase(cell.code)
                     : task::Phase::Completed;
    waiters = cell.link.wait_head;
    cell.link.wait_head = nullptr;
    waiters = OrderCompletionWaiters(waiters);
  }
  stripe.ready.notify_all();
  WakeCompletionWaiters(waiters);
  return task::Status::success();
}

task::Status CompletionPool::terminate(const CompletionLease lease,
                                       const ReasonCode code) noexcept {
  auto *const store = static_cast<Store *>(lease.authority);
  if (store == nullptr || !store->contains(lease.slot)) {
    return task::Status::fail(ReasonCode::TaskHandleStale);
  }
  Store::Cell &cell = store->at(lease.slot);
  Store::Stripe &stripe = store->stripe(lease.slot);
  CompletionWaiter *waiters = nullptr;
  {
    std::lock_guard lock{stripe.mutex};
    if (cell.generation != lease.generation ||
        cell.phase == task::Phase::Idle) {
      return task::Status::fail(ReasonCode::TaskHandleStale);
    }
    if (CompletionTerminal(cell.phase)) {
      return task::Status::fail(ReasonCode::TaskStateTransitionInvalid);
    }
    store->destroy_value(lease.slot, cell);
    cell.code = CompletionFailure(code);
    cell.outcome = CompletionOutcome::Failure;
    cell.phase = CompletionFailurePhase(cell.code);
    waiters = cell.link.wait_head;
    cell.link.wait_head = nullptr;
    waiters = OrderCompletionWaiters(waiters);
  }
  stripe.ready.notify_all();
  WakeCompletionWaiters(waiters);
  return task::Status::success();
}

void CompletionPool::release(const CompletionLease lease) noexcept {
  auto *const store = static_cast<Store *>(lease.authority);
  if (store == nullptr || !store->contains(lease.slot)) {
    return;
  }
  bool released = false;
  {
    std::lock_guard store_lock{store->mutex};
    Store::Cell &cell = store->at(lease.slot);
    std::lock_guard cell_lock{store->stripe(lease.slot).mutex};
    if (cell.generation != lease.generation ||
        !CompletionTerminal(cell.phase) || !cell.producer_live) {
      return;
    }
    cell.producer_live = false;
    if (cell.observers == 0u) {
      store->recycle(lease.slot, cell);
    }
    released = true;
  }
  if (released) {
    drop(store);
  }
}

void CompletionPool::release_observer(void *const authority,
                                      const std::uint32_t slot,
                                      const std::uint32_t generation) noexcept {
  auto *const store = static_cast<Store *>(authority);
  if (store == nullptr || !store->contains(slot)) {
    return;
  }
  bool released = false;
  {
    std::lock_guard store_lock{store->mutex};
    Store::Cell &cell = store->at(slot);
    std::lock_guard cell_lock{store->stripe(slot).mutex};
    if (cell.generation != generation || cell.phase == task::Phase::Idle ||
        cell.observers == 0u) {
      return;
    }
    --cell.observers;
    if (!cell.producer_live && cell.observers == 0u &&
        CompletionTerminal(cell.phase)) {
      store->recycle(slot, cell);
    }
    released = true;
  }
  if (released) {
    drop(store);
  }
}

} // namespace rund::node
