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
  return current.same_socket_object(expected) ? ::rund::ReasonCode::Ok
                                              : ::rund::ReasonCode::IoFdInvalid;
}

void InvalidateGeneration(SocketSlot &slot) noexcept {
  registry::retire(slot);
  registry::wait(slot);
  slot.identity = node::NativeFdIdentity::invalid();
}

} // namespace

SocketAdmission AdmitNativeSocket(const int native_socket) noexcept {
  if (native_socket < 0) {
    return SocketAdmission::failure(::rund::ReasonCode::IoFdInvalid);
  }
  const node::NativeFdIdentity identity =
      node::NativeDescribeFdIdentity(native_socket);
  if (identity.disposition() != node::NativeFdIdentityDisposition::Described) {
    return SocketAdmission::failure(::rund::ReasonCode::IoFdInvalid);
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
          return SocketAdmission::failure(::rund::ReasonCode::IoFdInvalid);
        }
        const std::uint64_t generation = registry::load(slot);
        if (!registry::active(generation) ||
            !slot.identity.same_socket_object(identity)) {
          reused_fd_needs_invalidation = true;
          needs_reservation =
              HasOwner(active_owner) && !SameOwner(slot, active_owner);
        } else {
          return SocketAdmission::failure(::rund::ReasonCode::TaskInvalid);
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
            if (!slot.hot.closing &&
                (!registry::active(registry::load(slot)) ||
                 !slot.identity.same_socket_object(identity))) {
              const node::NativeFdIdentity current =
                  node::NativeDescribeFdIdentity(native_socket);
              if (current.same_socket_object(identity)) {
                extra_release = TakeOwner(slot);
                InvalidateGeneration(slot);
              }
            }
          }
        }
        ReleaseRuntimeRegistryOwner(extra_release);
        return SocketAdmission::failure(
            ::rund::ReasonCode::TaskCapacityExceeded);
      }
      reserved = true;
      reserved_for_catch = true;
    }

    SocketRegistryOwner extra_release{};
    SocketAdmission result = [&]() -> SocketAdmission {
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
          return SocketAdmission::failure(::rund::ReasonCode::IoFdInvalid);
        } else if (!registry::active(generation) ||
                   !slot.identity.same_socket_object(identity)) {
          const node::NativeFdIdentity current =
              node::NativeDescribeFdIdentity(native_socket);
          if (!current.same_socket_object(identity)) {
            if (reserved) {
              extra_release = active_owner;
              reserved = false;
              reserved_for_catch = false;
            }
            return SocketAdmission::failure(::rund::ReasonCode::IoFdInvalid);
          }

          const ::rund::ReasonCode prepared =
              PrepareStableSocketSend(native_socket, identity);
          if (prepared != ::rund::ReasonCode::Ok) {
            extra_release = TakeOwner(slot);
            InvalidateGeneration(slot);
            return SocketAdmission::failure(prepared);
          }

          const std::uint64_t activated = registry::activate(slot);
          if (activated == 0u) {
            slot.identity = node::NativeFdIdentity::invalid();
            extra_release = TakeOwner(slot);
            sockets.release(slot);
            return SocketAdmission::failure(
                ::rund::ReasonCode::TaskCapacityExceeded);
          }

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
          return SocketAdmission::success(MakeAdmittedSocket(slot, activated));
        }

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
        return SocketAdmission::failure(::rund::ReasonCode::TaskInvalid);
      }

      const node::NativeFdIdentity current =
          node::NativeDescribeFdIdentity(native_socket);
      if (!current.same_socket_object(identity)) {
        if (reserved) {
          extra_release = active_owner;
          reserved = false;
          reserved_for_catch = false;
        }
        return SocketAdmission::failure(::rund::ReasonCode::IoFdInvalid);
      }

      const ::rund::ReasonCode prepared =
          PrepareStableSocketSend(native_socket, identity);
      if (prepared != ::rund::ReasonCode::Ok) {
        return SocketAdmission::failure(prepared);
      }

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
        return SocketAdmission::failure(
            ::rund::ReasonCode::TaskCapacityExceeded);
      }

      if (reserved) {
        AssignOwner(*bound, active_owner);
      }
      reserved = false;
      reserved_for_catch = false;
      return SocketAdmission::success(MakeAdmittedSocket(*bound, generation));
    }();
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
    return SocketAdmission::failure(::rund::ReasonCode::TaskCapacityExceeded);
  }
}

} // namespace rund::net
