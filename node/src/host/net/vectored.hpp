#pragma once

#include <rund/net/vectored.hpp>

#include "../../runtime/platform/net.hpp"

#include <cstdint>

namespace rund::net::batch::detail {

[[nodiscard]] ReceiveResult
complete_receive(std::uint64_t socket_id, std::span<const Buffer> slices,
                 std::uint64_t admitted_bytes,
                 const node::NativeCallResult &native) noexcept;

[[nodiscard]] SendResult
complete_send(std::uint64_t socket_id, std::span<const Slice> slices,
              std::uint64_t admitted_bytes,
              const node::NativeCallResult &native) noexcept;

} // namespace rund::net::batch::detail
