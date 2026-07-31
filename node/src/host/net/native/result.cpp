#include "result.hpp"

#include "../socket/access.hpp"

#include "../../../runtime/platform/io.hpp"

#include <algorithm>
namespace rund::net {

::rund::host::Status StatusForNative(const node::NativeCallResult &native) noexcept {
  switch (CodeForNative(native)) {
  case ::rund::ReasonCode::Ok:
    return ::rund::host::Status::Ok;
  case ::rund::ReasonCode::TaskInvalid:
    return ::rund::host::Status::Invalid;
  case ::rund::ReasonCode::IoUnsupported:
    return ::rund::host::Status::Unsupported;
  case ::rund::ReasonCode::IoWouldBlock:
    return ::rund::host::Status::WouldBlock;
  default:
    return ::rund::host::Status::SyscallFailed;
  }
}

::rund::ReasonCode CodeForNative(const node::NativeCallResult &native) noexcept {
  switch (native.state) {
  case node::NativeCallState::Complete:
  case node::NativeCallState::InProgress:
    return ::rund::ReasonCode::Ok;
  case node::NativeCallState::Failed:
    return ::rund::ReasonCode::IoSyscallFailed;
  case node::NativeCallState::InvalidInput:
    return ::rund::ReasonCode::TaskInvalid;
  case node::NativeCallState::Unsupported:
    return ::rund::ReasonCode::IoUnsupported;
  case node::NativeCallState::WouldBlock:
    return ::rund::ReasonCode::IoWouldBlock;
  }
  return ::rund::ReasonCode::IoSyscallFailed;
}

std::uint64_t CompletedByteCount(const node::NativeCallResult &native,
                                 const std::uint64_t requested) noexcept {
  if (native.value <= 0) {
    return 0u;
  }
  const std::uint64_t completed = static_cast<std::uint64_t>(native.value);
  return std::min(completed, requested);
}

} // namespace rund::net
