#pragma once

#include <rund/net/bytes.hpp>

#include <cstddef>
#include <span>

namespace rund::net::batch {

struct Slice {
  const std::byte *data = nullptr;
  std::size_t size = 0u;
};

struct Buffer {
  std::byte *data = nullptr;
  std::size_t size = 0u;
};

[[nodiscard]] ReceiveResult
receive(ready::Ticket &&ticket, std::span<const Buffer> slices) noexcept;
[[nodiscard]] SendResult
send(ready::Ticket &&ticket, std::span<const Slice> slices) noexcept;

} // namespace rund::net::batch
