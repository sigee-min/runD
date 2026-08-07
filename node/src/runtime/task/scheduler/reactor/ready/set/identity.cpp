#include "identity.hpp"

#include <limits>

namespace rund::node {
namespace {

constexpr std::uint64_t kExhaustedSlotId =
    std::numeric_limits<std::uint64_t>::max();

[[nodiscard]] constexpr bool Odd(const std::uint64_t value) noexcept {
  return (value & 1u) != 0u;
}

} // namespace

ReactorReadySetIdentityOwner::ReactorReadySetIdentityOwner(
    const std::uint64_t next_slot_id) noexcept
    : next_slot_id_(next_slot_id) {}

bool ReactorReadySetIdentityOwner::empty(
    const ::rund::net::ready::Set handle) noexcept {
  return handle.id == 0u && handle.generation == 0u;
}

bool ReactorReadySetIdentityOwner::valid(
    const ::rund::net::ready::Set handle) noexcept {
  return handle.id != 0u && handle.id != kExhaustedSlotId &&
         handle.generation != 0u;
}

bool ReactorReadySetIdentityOwner::same(
    const ::rund::net::ready::Set left,
    const ::rund::net::ready::Set right) noexcept {
  return left.id == right.id && left.generation == right.generation;
}

bool ReactorReadySetIdentityOwner::live(
    const ReactorReadySetIdentityState &state) noexcept {
  return state.live && valid(state.handle) && Odd(state.handle.generation);
}

bool ReactorReadySetIdentityOwner::matches(
    const ReactorReadySetIdentityState &state,
    const ::rund::net::ready::Set handle) noexcept {
  return live(state) && valid(handle) && same(state.handle, handle);
}

ReactorReadySetActivation ReactorReadySetIdentityOwner::activation(
    const ReactorReadySetIdentityState &state) noexcept {
  if (state.live) {
    return ReactorReadySetActivation::Invalid;
  }
  if (empty(state.handle)) {
    return ReactorReadySetActivation::NeedsSlotId;
  }
  if (!valid(state.handle)) {
    return ReactorReadySetActivation::Invalid;
  }
  if (state.handle.generation == kExhaustedSlotId) {
    return ReactorReadySetActivation::NeedsSlotId;
  }
  return Odd(state.handle.generation)
             ? ReactorReadySetActivation::Invalid
             : ReactorReadySetActivation::ReuseGeneration;
}

bool ReactorReadySetIdentityOwner::issue_slot_id(
    std::uint64_t *const slot_id) noexcept {
  if (slot_id == nullptr) {
    return false;
  }
  std::uint64_t current = next_slot_id_.load(std::memory_order_relaxed);
  while (current != 0u && current != kExhaustedSlotId) {
    if (next_slot_id_.compare_exchange_weak(current, current + 1u,
                                            std::memory_order_relaxed,
                                            std::memory_order_relaxed)) {
      *slot_id = current;
      return true;
    }
  }
  return false;
}

bool ReactorReadySetIdentityOwner::activate(
    ReactorReadySetIdentityState &state,
    ::rund::net::ready::Set *const handle) noexcept {
  if (handle == nullptr) {
    return false;
  }
  ::rund::net::ready::Set next{};
  switch (activation(state)) {
  case ReactorReadySetActivation::Invalid:
    return false;
  case ReactorReadySetActivation::ReuseGeneration:
    next = state.handle;
    ++next.generation;
    break;
  case ReactorReadySetActivation::NeedsSlotId:
    if (!issue_slot_id(&next.id)) {
      return false;
    }
    next.generation = 1u;
    break;
  }
  state.handle = next;
  state.live = true;
  *handle = next;
  return true;
}

bool ReactorReadySetIdentityOwner::retire(
    ReactorReadySetIdentityState &state, const ::rund::net::ready::Set expected,
    ::rund::net::ready::Set *const tombstone) noexcept {
  if (tombstone == nullptr || !matches(state, expected)) {
    return false;
  }
  ::rund::net::ready::Set next = state.handle;
  if (next.generation != kExhaustedSlotId) {
    ++next.generation;
  }
  state.handle = next;
  state.live = false;
  *tombstone = next;
  return true;
}

ReactorReadySetIdentityOwner &ProcessReadySetIdentityOwner() noexcept {
  static ReactorReadySetIdentityOwner owner{};
  return owner;
}

} // namespace rund::node
