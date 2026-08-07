#pragma once

#include <rund/net/bytes.hpp>

#include "../../runtime/platform/net.hpp"

#include <cstdint>

namespace rund::net::ready::detail {

struct Claim;

} // namespace rund::net::ready::detail

namespace rund::net::detail {

[[nodiscard]] ReceiveResult
receive_attempt(const ready::detail::Claim &claim,
                std::span<std::byte> buffer) noexcept;

[[nodiscard]] SendResult
send_attempt(const ready::detail::Claim &claim,
             std::span<const std::byte> buffer) noexcept;

[[nodiscard]] ReceiveResult
complete_receive(std::uint64_t socket_id, std::span<std::byte> buffer,
                 const node::NativeCallResult &native) noexcept;

[[nodiscard]] SendResult
complete_send(std::uint64_t socket_id, std::span<const std::byte> buffer,
              const node::NativeCallResult &native) noexcept;

} // namespace rund::net::detail
