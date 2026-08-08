#include "state.hpp"

#include "../socket/access.hpp"

#include <memory>

namespace rund::net {

namespace registry {
namespace {

void store(SocketSlot &slot, const std::uint64_t generation) noexcept {
  slot.hot.generation.store(generation, std::memory_order_release);
}

} // namespace

bool active(const std::uint64_t generation) noexcept {
  return generation != exhausted && (generation & 1u) != 0u;
}

std::uint64_t load(const SocketSlot &slot) noexcept {
  return slot.hot.generation.load(std::memory_order_acquire);
}

std::uint64_t activate(SocketSlot &slot) noexcept {
  if (slot.hot.closing) {
    return 0u;
  }
  const std::uint64_t current = load(slot);
  if (current == exhausted) {
    return 0u;
  }
  const std::uint64_t step = active(current) ? 2u : 1u;
  if (current >= exhausted - step) {
    store(slot, exhausted);
    return 0u;
  }
  const std::uint64_t next = current + step;
  store(slot, next);
  return next;
}

void retire(SocketSlot &slot) noexcept {
  const std::uint64_t current = load(slot);
  if (active(current)) {
    store(slot, current + 1u);
  }
}

bool retire(SocketSlot &slot, const std::uint64_t generation) noexcept {
  if (!active(generation)) {
    return false;
  }
  std::uint64_t expected = generation;
  return slot.hot.generation.compare_exchange_strong(expected, generation + 1u,
                                                     std::memory_order_acq_rel,
                                                     std::memory_order_acquire);
}

void wait(const SocketSlot &slot) noexcept {
  std::uint32_t count = slot.hot.readers.load(std::memory_order_acquire);
  while (count != 0u) {
    slot.hot.readers.wait(count, std::memory_order_acquire);
    count = slot.hot.readers.load(std::memory_order_acquire);
  }
}

} // namespace registry

std::shared_mutex &RegistryGate() noexcept {
  static std::shared_mutex mutex{};
  return mutex;
}

SocketRegistry &Registry() noexcept {
  static SocketRegistry sockets{};
  return sockets;
}

SocketSlot *SocketRegistry::find(const int native) noexcept {
  const auto found = live_.find(native);
  return found == live_.end() ? nullptr : found->second;
}

const SocketSlot *SocketRegistry::find(const int native) const noexcept {
  const auto found = live_.find(native);
  return found == live_.end() ? nullptr : found->second;
}

SocketSlot *SocketRegistry::bind(const int native,
                                 const node::NativeFdIdentity &identity) {
  if (find(native) != nullptr) {
    return nullptr;
  }

  SocketSlot *slot = free_;
  const bool reused = slot != nullptr;
  if (reused) {
    free_ = slot->next;
    slot->next = nullptr;
    --reusable_;
  } else {
    // Generation exhaustion permanently removes capacity. Allocating a
    // replacement here would make retained storage grow with history instead
    // of the admitted concurrency high-water.
    if (burned_ != 0u) {
      return nullptr;
    }
    auto owned = std::make_unique<SocketSlot>();
    slot = owned.get();
    owned->storage = std::move(slots_);
    slots_ = std::move(owned);
    ++slot_count_;
  }

  slot->hot.native.store(native, std::memory_order_relaxed);
  slot->identity = identity;
  slot->hot.readers.store(0u, std::memory_order_relaxed);
  slot->active_owner = {};
  slot->hot.closing = false;
  try {
    if (reused) {
      slot->index.key() = native;
      slot->index.mapped() = slot;
      auto inserted = live_.insert(std::move(slot->index));
      if (inserted.inserted) {
        return slot;
      }
      slot->index = std::move(inserted.node);
    } else {
      const auto inserted = live_.try_emplace(native, slot);
      if (inserted.second) {
        return slot;
      }
    }
  } catch (...) {
    slot->hot.native.store(-1, std::memory_order_relaxed);
    slot->identity = node::NativeFdIdentity::invalid();
    slot->next = free_;
    free_ = slot;
    ++reusable_;
    throw;
  }

  slot->hot.native.store(-1, std::memory_order_relaxed);
  slot->identity = node::NativeFdIdentity::invalid();
  slot->next = free_;
  free_ = slot;
  ++reusable_;
  return nullptr;
}

void SocketRegistry::release(SocketSlot &slot) noexcept {
  const auto found =
      live_.find(slot.hot.native.load(std::memory_order_relaxed));
  if (found == live_.end() || found->second != &slot) {
    return;
  }
  slot.index = live_.extract(found);
  slot.hot.native.store(-1, std::memory_order_release);
  slot.identity = node::NativeFdIdentity::invalid();
  slot.active_owner = {};
  slot.hot.closing = false;
  slot.next = nullptr;
  if (registry::load(slot) != registry::exhausted) {
    slot.next = free_;
    free_ = &slot;
    ++reusable_;
  } else {
    ++burned_;
  }
}

SocketRegistryStats SocketRegistry::stats() const noexcept {
  return SocketRegistryStats{
      .slots = slot_count_,
      .live = live_.size(),
      .reusable = reusable_,
      .burned = burned_,
  };
}

SocketRegistry::~SocketRegistry() noexcept {
  while (slots_ != nullptr) {
    std::unique_ptr<SocketSlot> next = std::move(slots_->storage);
    slots_ = std::move(next);
  }
}

Socket MakeAdmittedSocket(SocketSlot &slot,
                          const std::uint64_t generation) noexcept {
  return detail::SocketAccess::make(&slot, generation);
}

int detail::SocketAccess::native(const SocketView socket) noexcept {
  const SocketSlot *const slot = detail::SocketAccess::slot(socket);
  return slot == nullptr ? -1
                         : slot->hot.native.load(std::memory_order_acquire);
}

bool HasOwner(const SocketRegistryOwner &owner) noexcept {
  return owner.active();
}

bool SameOwner(const SocketSlot &slot,
               const SocketRegistryOwner &owner) noexcept {
  return HasOwner(owner) && slot.active_owner.external() == owner.external() &&
         (owner.external() ||
          slot.active_owner.scheduler_id == owner.scheduler_id);
}

void AssignOwner(SocketSlot &slot, const SocketRegistryOwner &owner) noexcept {
  slot.active_owner = owner;
}

SocketRegistryOwner TakeOwner(SocketSlot &slot) noexcept {
  SocketRegistryOwner owner = slot.active_owner;
  slot.active_owner = SocketRegistryOwner{};
  return owner;
}

void ReleaseRuntimeRegistryOwner(const SocketRegistryOwner &owner) noexcept {
  if (HasOwner(owner)) {
    SocketRegistryAccess::ReleaseOwner(owner);
  }
}

bool ReserveRuntimeRegistryOwner(const SocketRegistryOwner &owner) noexcept {
  if (!HasOwner(owner)) {
    return false;
  }
  return SocketRegistryAccess::TryAdmit(owner);
}

} // namespace rund::net
