#include "local.hpp"

#include "../../host.hpp"

#include <rund/task/handle/ref.hpp>

namespace rund::node {

CompletionWaiter *OrderCompletionWaiters(CompletionWaiter *waiter) noexcept {
  CompletionWaiter *ordered = nullptr;
  while (waiter != nullptr) {
    CompletionWaiter *const next = waiter->next;
    waiter->next = ordered;
    waiter->previous = nullptr;
    waiter->state.store(CompletionWaiterState::Waking,
                        std::memory_order_release);
    ordered = waiter;
    waiter = next;
  }
  return ordered;
}

void WakeCompletionWaiters(CompletionWaiter *ordered) noexcept {
  while (ordered != nullptr) {
    CompletionWaiter *const next = ordered->next;
    const auto wake = ordered->wake;
    void *const value = ordered->value;
    ordered->next = nullptr;
    ordered->previous = nullptr;
    ordered->state.store(CompletionWaiterState::Idle,
                         std::memory_order_release);
    if (wake != nullptr) {
      wake(value);
    }
    ordered = next;
  }
}

task::Poll
CompletionPool::observer_poll(void *const authority, const std::uint32_t slot,
                              const std::uint32_t generation) noexcept {
  return poll(CompletionLease{
      .authority = authority, .slot = slot, .generation = generation});
}

task::Status
CompletionPool::observer_wait(void *const authority, const std::uint32_t slot,
                              const std::uint32_t generation) noexcept {
  if (scheduler_host::ActiveTask()) {
    return task::Status::fail(ReasonCode::TaskWorkerWaitForbidden);
  }
  return wait(CompletionLease{
      .authority = authority, .slot = slot, .generation = generation});
}

task::Status
CompletionPool::observer_copy(void *const authority, const std::uint32_t slot,
                              const std::uint32_t generation, void *const out,
                              const void *const type, const CopyFn copy) {
  return copy_raw(CompletionLease{.authority = authority,
                                  .slot = slot,
                                  .generation = generation},
                  out, type, copy);
}

task::Status CompletionPool::wait(const CompletionLease lease) noexcept {
  auto *const store = static_cast<Store *>(lease.authority);
  if (store == nullptr || !store->contains(lease.slot)) {
    return task::Status::fail(ReasonCode::TaskHandleStale);
  }
  Store::Cell &cell = store->at(lease.slot);
  Store::Stripe &stripe = store->stripe(lease.slot);
  std::unique_lock lock{stripe.mutex};
  if (cell.generation != lease.generation || cell.phase == task::Phase::Idle) {
    return task::Status::fail(ReasonCode::TaskHandleStale);
  }
  stripe.ready.wait(lock, [&] {
    return cell.generation != lease.generation ||
           cell.phase == task::Phase::Idle || CompletionTerminal(cell.phase);
  });
  if (cell.generation != lease.generation || cell.phase == task::Phase::Idle) {
    return task::Status::fail(ReasonCode::TaskHandleStale);
  }
  return cell.code == ReasonCode::Ok ? task::Status::success()
                                     : task::Status::fail(cell.code);
}

task::Poll
CompletionPool::wait_for(const ::rund::detail::task::ResultRef &observer,
                         const std::chrono::nanoseconds timeout) noexcept {
  if (observer.poll != observer_poll || observer.wait != observer_wait ||
      observer.copy != observer_copy || observer.release != release_observer) {
    return task::Poll{.phase = task::Phase::Failed,
                      .code = ReasonCode::TaskHandleStale};
  }
  const CompletionLease lease{.authority = observer.authority,
                              .slot = observer.slot,
                              .generation = observer.generation};
  if (timeout <= std::chrono::nanoseconds::zero()) {
    return poll(lease);
  }
  auto *const store = static_cast<Store *>(lease.authority);
  if (store == nullptr || !store->contains(lease.slot)) {
    return task::Poll{.phase = task::Phase::Failed,
                      .code = ReasonCode::TaskHandleStale};
  }
  Store::Cell &cell = store->at(lease.slot);
  Store::Stripe &stripe = store->stripe(lease.slot);
  std::unique_lock lock{stripe.mutex};
  stripe.ready.wait_for(lock, timeout, [&] {
    return cell.generation != lease.generation ||
           cell.phase == task::Phase::Idle || CompletionTerminal(cell.phase);
  });
  if (cell.generation != lease.generation || cell.phase == task::Phase::Idle) {
    return task::Poll{.phase = task::Phase::Failed,
                      .code = ReasonCode::TaskHandleStale};
  }
  return task::Poll{.phase = cell.phase, .code = cell.code};
}

bool CompletionPool::park(const CompletionLease lease,
                          CompletionWaiter &waiter) noexcept {
  auto *const store = static_cast<Store *>(lease.authority);
  if (store == nullptr || !store->contains(lease.slot) ||
      waiter.wake == nullptr ||
      waiter.state.load(std::memory_order_acquire) !=
          CompletionWaiterState::Idle) {
    return false;
  }
  Store::Cell &cell = store->at(lease.slot);
  std::lock_guard lock{store->stripe(lease.slot).mutex};
  if (cell.generation != lease.generation || cell.phase == task::Phase::Idle ||
      CompletionTerminal(cell.phase) ||
      waiter.state.load(std::memory_order_relaxed) !=
          CompletionWaiterState::Idle) {
    return false;
  }
  waiter.previous = nullptr;
  waiter.next = cell.link.wait_head;
  if (waiter.next != nullptr) {
    waiter.next->previous = &waiter;
  }
  waiter.state.store(CompletionWaiterState::Linked, std::memory_order_release);
  cell.link.wait_head = &waiter;
  return true;
}

bool CompletionPool::unpark(const CompletionLease lease,
                            CompletionWaiter &waiter) noexcept {
  auto *const store = static_cast<Store *>(lease.authority);
  if (store == nullptr || !store->contains(lease.slot) ||
      waiter.state.load(std::memory_order_acquire) !=
          CompletionWaiterState::Linked) {
    return false;
  }
  Store::Cell &cell = store->at(lease.slot);
  std::lock_guard lock{store->stripe(lease.slot).mutex};
  if (cell.generation != lease.generation || cell.phase == task::Phase::Idle ||
      waiter.state.load(std::memory_order_relaxed) !=
          CompletionWaiterState::Linked) {
    return false;
  }
  if (waiter.previous == nullptr) {
    if (cell.link.wait_head != &waiter) {
      return false;
    }
    cell.link.wait_head = waiter.next;
  } else {
    waiter.previous->next = waiter.next;
  }
  if (waiter.next != nullptr) {
    waiter.next->previous = waiter.previous;
  }
  waiter.next = nullptr;
  waiter.previous = nullptr;
  waiter.state.store(CompletionWaiterState::Idle, std::memory_order_release);
  return true;
}

} // namespace rund::node
