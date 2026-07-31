#include "../alignment.hpp"
#include "local.hpp"

#include <rund/task/handle/typed.hpp>

#include <new>

namespace rund::node {

CompletionPool::Store::~Store() {
  const std::uint32_t resident = created.load(std::memory_order_acquire);
  for (std::uint32_t slot = 0u; slot < resident; ++slot) {
    Cell &cell = at(slot);
    destroy_value(slot, cell);
    ::operator delete(cell.result, std::align_val_t{limits.result_alignment});
  }
  for (std::uint32_t page = 0u; page < page_count; ++page) {
    delete[] pages[page].load(std::memory_order_acquire);
  }
}

void *CompletionPool::Store::result(const std::uint32_t slot) const noexcept {
  return static_cast<std::byte *>(at(slot).result) + limits.result_alignment;
}

CompletionPool::Store::Cell &
CompletionPool::Store::at(const std::uint32_t slot) const noexcept {
  Cell *const page =
      pages[slot / kCompletionPageSlots].load(std::memory_order_acquire);
  return page[slot % kCompletionPageSlots];
}

bool CompletionPool::Store::contains(const std::uint32_t slot) const noexcept {
  return slot < created.load(std::memory_order_acquire);
}

bool CompletionPool::Store::grow() noexcept {
  const std::uint32_t first = created.load(std::memory_order_relaxed);
  if (first >= limits.capacity) {
    return false;
  }
  const std::uint32_t count =
      std::min(kCompletionPageSlots, limits.capacity - first);
  std::unique_ptr<Cell[]> page{new (std::nothrow) Cell[count]};
  if (!page) {
    return false;
  }
  for (std::uint32_t offset = count; offset != 0u; --offset) {
    const std::uint32_t slot = first + offset - 1u;
    Cell &cell = page[offset - 1u];
    cell.link.free_next = free_head;
    free_head = slot;
  }
  const std::uint32_t page_index = first / kCompletionPageSlots;
  pages[page_index].store(page.release(), std::memory_order_release);
  created.store(first + count, std::memory_order_release);
  return true;
}

CompletionPool::Store::ResultHeader &
CompletionPool::Store::result_header(Cell &cell) const noexcept {
  return *static_cast<ResultHeader *>(cell.result);
}

CompletionPool::Store::Stripe &
CompletionPool::Store::stripe(const std::uint32_t slot) const noexcept {
  return stripes[slot % stripe_count];
}

void CompletionPool::Store::destroy_value(const std::uint32_t slot,
                                          Cell &cell) noexcept {
  if (cell.outcome != CompletionOutcome::Value) {
    return;
  }
  result_header(cell).destroy(result(slot));
  result_header(cell) = {};
  cell.outcome = CompletionOutcome::Pending;
}

void CompletionPool::Store::recycle(const std::uint32_t slot,
                                    Cell &cell) noexcept {
  destroy_value(slot, cell);
  cell.code = ReasonCode::Ok;
  cell.phase = task::Phase::Idle;
  cell.outcome = CompletionOutcome::Pending;
  cell.producer_live = false;
  cell.link.wait_head = nullptr;
  if (cell.generation == std::numeric_limits<std::uint32_t>::max()) {
    return;
  }
  ++cell.generation;
  cell.link.free_next = free_head;
  free_head = slot;
}

CompletionPool::~CompletionPool() { reset(); }

void CompletionPool::drop(Store *const store) noexcept {
  if (store != nullptr &&
      store->refs.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
    delete store;
  }
}

task::Status CompletionPool::configure(const CompletionLimits limits) noexcept {
  reset();
  if (limits.capacity == 0u || limits.result_bytes == 0u ||
      limits.result_alignment < alignof(std::max_align_t) ||
      !alignment::power(limits.result_alignment)) {
    return task::Status::fail(ReasonCode::TaskCompletionCapacity);
  }
  Store *const store = new (std::nothrow) Store{};
  if (store == nullptr) {
    return task::Status::fail(ReasonCode::TaskCompletionCapacity);
  }
  store->limits = limits;
  store->stripe_count = std::min(limits.capacity, kCompletionStripeCount);
  store->stripes.reset(new (std::nothrow) Store::Stripe[store->stripe_count]);
  store->page_count =
      (limits.capacity + kCompletionPageSlots - 1u) / kCompletionPageSlots;
  store->pages.reset(new (std::nothrow)
                         std::atomic<Store::Cell *>[store->page_count]);
  if (!store->stripes || !store->pages) {
    delete store;
    return task::Status::fail(ReasonCode::TaskCompletionCapacity);
  }
  for (std::uint32_t page = 0u; page < store->page_count; ++page) {
    store->pages[page].store(nullptr, std::memory_order_relaxed);
  }
  store_ = store;
  return task::Status::success();
}

void CompletionPool::reset() noexcept {
  Store *const store = store_;
  store_ = nullptr;
  if (store == nullptr) {
    return;
  }
  {
    std::lock_guard lock{store->mutex};
    store->retired = true;
  }
  drop(store);
}

std::uint32_t CompletionPool::resident_cells() const noexcept {
  Store *const store = store_;
  if (store == nullptr) {
    return 0u;
  }
  std::lock_guard lock{store->mutex};
  return store->created.load(std::memory_order_acquire);
}

CompletionLease CompletionPool::claim() noexcept {
  Store *const store = store_;
  if (store == nullptr) {
    return {};
  }
  std::lock_guard lock{store->mutex};
  if (store->retired ||
      (store->free_head == kNoCompletionSlot && !store->grow())) {
    return {};
  }
  const std::uint32_t slot = store->free_head;
  Store::Cell &cell = store->at(slot);
  store->free_head = cell.link.free_next;
  {
    std::lock_guard cell_lock{store->stripe(slot).mutex};
    cell.phase = task::Phase::Admitted;
    cell.code = ReasonCode::Ok;
    cell.outcome = CompletionOutcome::Pending;
    cell.observers = 0u;
    cell.producer_live = true;
    cell.link.wait_head = nullptr;
  }
  store->refs.fetch_add(1u, std::memory_order_relaxed);
  return CompletionLease{
      .authority = store, .slot = slot, .generation = cell.generation};
}

CompletionLease
CompletionPool::lease(const CompletionSlot slot) const noexcept {
  return slot && store_ != nullptr
             ? CompletionLease{.authority = store_,
                               .slot = slot.slot,
                               .generation = slot.generation}
             : CompletionLease{};
}

CompletionSlot CompletionPool::slot(const CompletionLease lease) noexcept {
  return lease ? CompletionSlot{.slot = lease.slot,
                                .generation = lease.generation}
               : CompletionSlot{};
}

bool CompletionPool::retain_observer(const CompletionLease lease) noexcept {
  auto *const store = static_cast<Store *>(lease.authority);
  if (store == nullptr || !store->contains(lease.slot)) {
    return false;
  }
  Store::Cell &cell = store->at(lease.slot);
  std::lock_guard lock{store->stripe(lease.slot).mutex};
  if (cell.generation != lease.generation || cell.phase == task::Phase::Idle) {
    return false;
  }
  if (cell.observers == std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  ++cell.observers;
  store->refs.fetch_add(1u, std::memory_order_relaxed);
  return true;
}

::rund::detail::task::ResultRef
CompletionPool::observe_ref(const CompletionLease lease) noexcept {
  if (!retain_observer(lease)) {
    return {};
  }
  return ::rund::detail::task::ResultRef{.authority = lease.authority,
                                         .slot = lease.slot,
                                         .generation = lease.generation,
                                         .poll = &observer_poll,
                                         .wait = &observer_wait,
                                         .copy = &observer_copy,
                                         .release = &release_observer};
}

} // namespace rund::node
