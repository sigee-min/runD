#include "state.hpp"

#include "../../../runtime/platform/net.hpp"
#include "../native/result.hpp"

namespace rund::net {
namespace {

[[nodiscard]] ::rund::ReasonCode
PrepareStableSocketSend(const int native_socket,
                        const node::NativeFdIdentity &expected) noexcept {
  const node::NativeCallResult prepared =
      node::NativePrepareSocketSend(native_socket);
  if (prepared.state != node::NativeCallState::Complete) {
    const ::rund::ReasonCode code = CodeForNative(prepared);
    return code == ::rund::ReasonCode::Ok ? ::rund::ReasonCode::IoSyscallFailed
                                          : code;
  }
  const node::NativeFdIdentity current =
      node::NativeDescribeFdIdentity(native_socket);
  return current.ok && SameIdentity(current, expected)
             ? ::rund::ReasonCode::Ok
             : ::rund::ReasonCode::IoFdInvalid;
}

void InvalidateGeneration(SocketSlot &slot) noexcept {
  registry::retire(slot);
  registry::wait(slot);
  slot.identity = node::NativeFdIdentity{};
}

} // namespace

SocketAdmission AdmitNativeSocketImpl(const int native_socket) noexcept {
  if (native_socket < 0) {
    return SocketAdmission{.code = ::rund::ReasonCode::IoFdInvalid};
  }
  const node::NativeFdIdentity identity =
      node::NativeDescribeFdIdentity(native_socket);
  if (!identity.ok) {
    return SocketAdmission{.code = ::rund::ReasonCode::IoFdInvalid};
  }
  const SocketRegistryOwner active_owner = SocketRegistryAccess::ActiveOwner();
  bool needs_reservation = false;
  bool reused_fd_needs_invalidation = false;
  bool reserved_for_catch = false;
  try {
    {
      std::lock_guard lock{RegistryGate()};
      SocketRegistry &sockets = Registry();
      SocketSlot *const found = sockets.find(native_socket);
      if (found != nullptr) {
        SocketSlot &slot = *found;
        if (slot.hot.closing) {
          return SocketAdmission{.code = ::rund::ReasonCode::IoFdInvalid};
        }
        const std::uint64_t generation = registry::load(slot);
        if (!registry::active(generation) ||
            !SameIdentity(slot.identity, identity)) {
          reused_fd_needs_invalidation = true;
          needs_reservation =
              HasOwner(active_owner) && !SameOwner(slot, active_owner);
        } else {
          return SocketAdmission{.code = ::rund::ReasonCode::TaskInvalid};
        }
      } else {
        needs_reservation = HasOwner(active_owner);
      }
    }

    bool reserved = false;
    if (needs_reservation) {
      if (!ReserveRuntimeRegistryOwner(active_owner)) {
        SocketRegistryOwner extra_release{};
        if (reused_fd_needs_invalidation) {
          std::lock_guard lock{RegistryGate()};
          SocketRegistry &sockets = Registry();
          SocketSlot *const found = sockets.find(native_socket);
          if (found != nullptr) {
            SocketSlot &slot = *found;
            if (!slot.hot.closing && (!registry::active(registry::load(slot)) ||
                                  !SameIdentity(slot.identity, identity))) {
              const node::NativeFdIdentity current =
                  node::NativeDescribeFdIdentity(native_socket);
              if (current.ok && SameIdentity(current, identity)) {
                extra_release = TakeOwner(slot);
                InvalidateGeneration(slot);
              }
            }
          }
        }
        ReleaseRuntimeRegistryOwner(extra_release);
        return SocketAdmission{.code =
                                   ::rund::ReasonCode::TaskCapacityExceeded};
      }
      reserved = true;
      reserved_for_catch = true;
    }

    SocketAdmission result{};
    SocketRegistryOwner extra_release{};
    {
      std::lock_guard lock{RegistryGate()};
      SocketRegistry &sockets = Registry();
      SocketSlot *const found = sockets.find(native_socket);
      if (found != nullptr) {
        SocketSlot &slot = *found;
        const std::uint64_t generation = registry::load(slot);
        if (slot.hot.closing) {
          if (reserved) {
            extra_release = active_owner;
            reserved = false;
            reserved_for_catch = false;
          }
          result = SocketAdmission{.code = ::rund::ReasonCode::IoFdInvalid};
        } else if (!registry::active(generation) ||
                   !SameIdentity(slot.identity, identity)) {
          const node::NativeFdIdentity current =
              node::NativeDescribeFdIdentity(native_socket);
          if (!current.ok || !SameIdentity(current, identity)) {
            if (reserved) {
              extra_release = active_owner;
              reserved = false;
              reserved_for_catch = false;
            }
            result = SocketAdmission{.code = ::rund::ReasonCode::IoFdInvalid};
          } else {
            const ::rund::ReasonCode prepared =
                PrepareStableSocketSend(native_socket, identity);
            if (prepared != ::rund::ReasonCode::Ok) {
              extra_release = TakeOwner(slot);
              InvalidateGeneration(slot);
              result = SocketAdmission{.code = prepared};
            } else {
              const std::uint64_t activated = registry::activate(slot);
              if (activated == 0u) {
                slot.identity = node::NativeFdIdentity{};
                extra_release = TakeOwner(slot);
                sockets.release(slot);
                result = SocketAdmission{
                    .code = ::rund::ReasonCode::TaskCapacityExceeded};
              } else {
                if (!HasOwner(active_owner)) {
                  extra_release = TakeOwner(slot);
                } else if (!SameOwner(slot, active_owner)) {
                  extra_release = TakeOwner(slot);
                  if (reserved) {
                    AssignOwner(slot, active_owner);
                    reserved = false;
                    reserved_for_catch = false;
                  }
                }
                slot.identity = identity;
                result = SocketAdmission{
                    .socket = MakeAdmittedSocket(slot, activated),
                    .code = ::rund::ReasonCode::Ok,
                };
              }
            }
          }
        } else {
          if (reserved && !SameOwner(slot, active_owner)) {
            extra_release = TakeOwner(slot);
            AssignOwner(slot, active_owner);
            reserved = false;
            reserved_for_catch = false;
          } else if (reserved) {
            extra_release = active_owner;
            reserved = false;
            reserved_for_catch = false;
          }
          result = SocketAdmission{.code = ::rund::ReasonCode::TaskInvalid};
        }
      } else {
        const node::NativeFdIdentity current =
            node::NativeDescribeFdIdentity(native_socket);
        if (!current.ok || !SameIdentity(current, identity)) {
          if (reserved) {
            extra_release = active_owner;
            reserved = false;
            reserved_for_catch = false;
          }
          result = SocketAdmission{.code = ::rund::ReasonCode::IoFdInvalid};
        } else {
          const ::rund::ReasonCode prepared =
              PrepareStableSocketSend(native_socket, identity);
          if (prepared != ::rund::ReasonCode::Ok) {
            result = SocketAdmission{.code = prepared};
          } else {
            SocketSlot *const bound = sockets.bind(native_socket, identity);
            const std::uint64_t generation =
                bound == nullptr ? 0u : registry::activate(*bound);
            if (bound == nullptr || generation == 0u) {
              if (bound != nullptr) {
                sockets.release(*bound);
              }
              if (reserved) {
                extra_release = active_owner;
                reserved = false;
                reserved_for_catch = false;
              }
              result = SocketAdmission{
                  .code = ::rund::ReasonCode::TaskCapacityExceeded};
            } else {
              if (reserved) {
                AssignOwner(*bound, active_owner);
              }
              reserved = false;
              reserved_for_catch = false;
              result = SocketAdmission{
                  .socket = MakeAdmittedSocket(*bound, generation),
                  .code = ::rund::ReasonCode::Ok,
              };
            }
          }
        }
      }
    }
    ReleaseRuntimeRegistryOwner(extra_release);
    if (reserved) {
      ReleaseRuntimeRegistryOwner(active_owner);
      reserved_for_catch = false;
    }
    return result;
  } catch (...) {
    if (reserved_for_catch) {
      ReleaseRuntimeRegistryOwner(active_owner);
    }
    return SocketAdmission{.code = ::rund::ReasonCode::TaskCapacityExceeded};
  }
}

SocketAdmission AdmitNativeSocket(const int native_socket) noexcept {
  return AdmitNativeSocketImpl(native_socket);
}

} // namespace rund::net
