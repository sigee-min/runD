#pragma once

#include <rund/net/result.hpp>

#include <cstdint>

namespace rund::net {

struct Limits : Status {
  using Status::Status;

  std::uint32_t ready_sets = 0u;
  std::uint32_t ready_set_members = 0u;
  std::uint32_t max_ready_sets = 0u;
  std::uint32_t max_ready_set_members = 0u;
  std::uint32_t max_iov = 0u;
  std::uint32_t max_datagram_bytes = 0u;
  std::uint32_t max_socket_registry_entries = 0u;
};

[[nodiscard]] Limits limits() noexcept;

} // namespace rund::net
