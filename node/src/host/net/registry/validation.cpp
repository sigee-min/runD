#include "state.hpp"

#include <cstdint>
#include <limits>
#include <utility>

namespace rund::net {

bool IsCurrentSocket(const SocketView socket) noexcept {
  SocketSlot *const slot = detail::SocketAccess::slot(socket);
  if (slot == nullptr || detail::SocketAccess::generation(socket) == 0u) {
    return false;
  }
  const std::uint64_t generation = registry::load(*slot);
  return registry::active(generation) &&
         generation == detail::SocketAccess::generation(socket);
}

SocketLease::~SocketLease() noexcept { release(); }

SocketLease::SocketLease(SocketLease &&other) noexcept
    : slot_(std::exchange(other.slot_, nullptr)),
      native_(std::exchange(other.native_, -1)) {}

SocketLease &SocketLease::operator=(SocketLease &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  slot_ = std::exchange(other.slot_, nullptr);
  native_ = std::exchange(other.native_, -1);
  return *this;
}

void SocketLease::release() noexcept {
  if (slot_ == nullptr) {
    return;
  }
  std::atomic_ref<std::uint32_t> readers{slot_->hot.readers};
  const std::uint32_t previous =
      readers.fetch_sub(1u, std::memory_order_acq_rel);
  if (previous == 1u) {
    readers.notify_all();
  }
  slot_ = nullptr;
  native_ = -1;
}

SocketLease LeaseSocket(const SocketView socket) noexcept {
  SocketSlot *const slot = detail::SocketAccess::slot(socket);
  const std::uint64_t generation = detail::SocketAccess::generation(socket);
  if (slot == nullptr || !registry::active(generation) ||
      registry::load(*slot) != generation) {
    return {};
  }

  std::atomic_ref<std::uint32_t> readers{slot->hot.readers};
  std::uint32_t count = readers.load(std::memory_order_relaxed);
  while (count != std::numeric_limits<std::uint32_t>::max()) {
    if (readers.compare_exchange_weak(count, count + 1u,
                                      std::memory_order_acq_rel,
                                      std::memory_order_relaxed)) {
      if (registry::load(*slot) == generation) {
        return SocketLease{slot, slot->hot.native.load(std::memory_order_relaxed)};
      }
      const std::uint32_t previous =
          readers.fetch_sub(1u, std::memory_order_acq_rel);
      if (previous == 1u) {
        readers.notify_all();
      }
      return {};
    }
  }
  return {};
}

} // namespace rund::net
