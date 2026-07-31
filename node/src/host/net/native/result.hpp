#pragma once

#include <rund/host/event.hpp>
#include <rund/net/socket.hpp>
#include <rund/task/results.hpp>

#include "../../../runtime/platform/net.hpp"

#include <cstdint>

namespace rund::net {

[[nodiscard]] ::rund::host::Status
StatusForNative(const node::NativeCallResult &native) noexcept;
[[nodiscard]] ::rund::ReasonCode CodeForNative(const node::NativeCallResult &native) noexcept;
[[nodiscard]] std::uint64_t
CompletedByteCount(const node::NativeCallResult &native,
                   std::uint64_t requested) noexcept;

} // namespace rund::net
