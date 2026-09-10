#include "local.hpp"

#include "../../../../reactor/readiness/mask.hpp"

namespace rund::node::reactor_registry_detail {
namespace {

[[nodiscard]] bool RemoveInterest(ReactorFdState &state,
                                  const ReactorInterest interest) noexcept {
  if ((HasReactorInterest(interest, ReactorInterest::Read) &&
       state.read_count == 0u) ||
      (HasReactorInterest(interest, ReactorInterest::Write) &&
       state.write_count == 0u)) {
    return false;
  }
  if (HasReactorInterest(interest, ReactorInterest::Read)) {
    --state.read_count;
  }
  if (HasReactorInterest(interest, ReactorInterest::Write)) {
    --state.write_count;
  }
  return true;
}

} // namespace

void AddInterest(ReactorFdState &state,
                 const ReactorInterest interest) noexcept {
  if (HasReactorInterest(interest, ReactorInterest::Read)) {
    ++state.read_count;
  }
  if (HasReactorInterest(interest, ReactorInterest::Write)) {
    ++state.write_count;
  }
}

[[nodiscard]] ReactorInterest Interest(const ReactorFdState &state) noexcept {
  ReactorInterest interest = ReactorInterest::None;
  if (state.read_count != 0u) {
    interest = interest | ReactorInterest::Read;
  }
  if (state.write_count != 0u) {
    interest = interest | ReactorInterest::Write;
  }
  return interest;
}

void ResetSlot(ReactorRegistry &registry, const std::uint32_t index) noexcept {
  ReactorWaitSlot &slot = registry.slots[index];
  slot.wait = ReactorWait{};
  slot.previous_fd = kNoReactorSlot;
  slot.next_fd = kNoReactorSlot;
}

void ReleaseSlot(ReactorRegistry &registry,
                 const std::uint32_t index) noexcept {
  ResetSlot(registry, index);
  registry.free_slots.push_back(index);
  --registry.live;
}

[[nodiscard]] bool Unlink(ReactorRegistry &registry, ReactorFdState &fd,
                          const std::uint32_t index) noexcept {
  ReactorWaitSlot &slot = registry.slots[index];
  if (slot.wait.wait_id == 0u || slot.wait.fd != fd.fd || fd.wait_count == 0u) {
    return false;
  }
  if (slot.previous_fd == kNoReactorSlot) {
    if (fd.first_wait != index) {
      return false;
    }
  } else if (slot.previous_fd >= registry.slots.size() ||
             registry.slots[slot.previous_fd].next_fd != index) {
    return false;
  }
  if (slot.next_fd == kNoReactorSlot) {
    if (fd.last_wait != index) {
      return false;
    }
  } else if (slot.next_fd >= registry.slots.size() ||
             registry.slots[slot.next_fd].previous_fd != index) {
    return false;
  }
  if (!RemoveInterest(fd, slot.wait.interest)) {
    return false;
  }
  if (slot.previous_fd == kNoReactorSlot) {
    fd.first_wait = slot.next_fd;
  } else {
    registry.slots[slot.previous_fd].next_fd = slot.next_fd;
  }
  if (slot.next_fd == kNoReactorSlot) {
    fd.last_wait = slot.previous_fd;
  } else {
    registry.slots[slot.next_fd].previous_fd = slot.previous_fd;
  }
  --fd.wait_count;
  slot.previous_fd = kNoReactorSlot;
  slot.next_fd = kNoReactorSlot;
  return true;
}

} // namespace rund::node::reactor_registry_detail
