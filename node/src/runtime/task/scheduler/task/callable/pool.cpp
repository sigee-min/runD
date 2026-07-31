#include "pool.hpp"

#include <algorithm>
#include <new>

namespace rund::node {
namespace {

constexpr std::uint32_t kCallablePageSlots = 256u;

} // namespace

CallablePool::~CallablePool() { reset(); }

bool CallablePool::configure(const std::uint32_t capacity) noexcept {
  reset();
  if (capacity == 0u) {
    return false;
  }
  try {
    pages_.reserve((capacity + kCallablePageSlots - 1u) / kCallablePageSlots);
  } catch (...) {
    return false;
  }
  capacity_ = capacity;
  return true;
}

void CallablePool::reset() noexcept {
  for (const Page page : pages_) {
    for (std::uint32_t index = 0u; index < page.count; ++index) {
      Slot &slot = page.slots[index];
      if (slot.state == SlotState::Live) {
        reinterpret_cast<::rund::detail::task::Callable *>(slot.storage)
            ->~Callable();
      }
      slot.state = SlotState::Free;
    }
    delete[] page.slots;
  }
  pages_.clear();
  free_ = nullptr;
  capacity_ = 0u;
  created_ = 0u;
  live_ = 0u;
}

bool CallablePool::grow() noexcept {
  if (created_ >= capacity_) {
    return false;
  }
  const std::uint32_t count =
      std::min(kCallablePageSlots, capacity_ - created_);
  Slot *const slots = new (std::nothrow) Slot[count];
  if (slots == nullptr) {
    return false;
  }
  try {
    pages_.push_back(Page{.slots = slots, .count = count});
  } catch (...) {
    delete[] slots;
    return false;
  }
  for (std::uint32_t index = count; index != 0u; --index) {
    Slot &slot = slots[index - 1u];
    slot.next = free_;
    free_ = &slot;
  }
  created_ += count;
  return true;
}

::rund::detail::task::Callable *
CallablePool::claim(::rund::detail::task::Callable &&value) noexcept {
  if (free_ == nullptr && !grow()) {
    return nullptr;
  }
  Slot *const slot = free_;
  free_ = slot->next;
  slot->next = nullptr;
  slot->state = SlotState::Live;
  ++live_;
  return new (slot->storage)::rund::detail::task::Callable(std::move(value));
}

void CallablePool::destroy(
    ::rund::detail::task::Callable *const value) noexcept {
  if (value == nullptr) {
    return;
  }
  auto *const slot = reinterpret_cast<Slot *>(
      reinterpret_cast<std::byte *>(value) - offsetof(Slot, storage));
  if (slot->state != SlotState::Live) {
    return;
  }
  value->~Callable();
  slot->state = SlotState::Destroyed;
}

void CallablePool::release(
    ::rund::detail::task::Callable *const value) noexcept {
  if (value == nullptr) {
    return;
  }
  auto *const slot = reinterpret_cast<Slot *>(
      reinterpret_cast<std::byte *>(value) - offsetof(Slot, storage));
  if (slot->state == SlotState::Free) {
    return;
  }
  if (slot->state == SlotState::Live) {
    value->~Callable();
  }
  slot->state = SlotState::Free;
  slot->next = free_;
  free_ = slot;
  if (live_ != 0u) {
    --live_;
  }
}

} // namespace rund::node
