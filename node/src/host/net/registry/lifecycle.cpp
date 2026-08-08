#include "state.hpp"

namespace rund::net {

bool BeginSocketClose(const SocketView socket) noexcept {
  SocketSlot *const slot = detail::SocketAccess::slot(socket);
  if (slot == nullptr || detail::SocketAccess::generation(socket) == 0u) {
    return false;
  }
  {
    std::lock_guard lock{RegistryGate()};
    if (slot->hot.closing ||
        !registry::retire(*slot, detail::SocketAccess::generation(socket))) {
      return false;
    }
    slot->hot.closing = true;
    slot->identity = node::NativeFdIdentity::invalid();
  }
  registry::wait(*slot);
  return true;
}

void FinishSocketClose(const SocketView socket) noexcept {
  SocketSlot *const slot = detail::SocketAccess::slot(socket);
  if (slot == nullptr) {
    return;
  }
  SocketRegistryOwner owner{};
  {
    std::lock_guard lock{RegistryGate()};
    if (slot->hot.closing &&
        registry::load(*slot) ==
            detail::SocketAccess::generation(socket) + 1u) {
      owner = TakeOwner(*slot);
      Registry().release(*slot);
    }
  }
  ReleaseRuntimeRegistryOwner(owner);
}

} // namespace rund::net
