#pragma once

#include "../net.hpp"

#include <sys/socket.h>

namespace rund::node {

[[nodiscard]] bool PosixAddress(const ::rund::net::Address& address,
                                sockaddr_storage *storage,
                                socklen_t *length) noexcept;
[[nodiscard]] ::rund::net::Address AddressFromPosix(const sockaddr *address,
                                            socklen_t length) noexcept;

} // namespace rund::node
