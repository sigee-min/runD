#pragma once

#include <rund/net/result.hpp>

#include <cstdint>
#include <string_view>

namespace rund::net::frame {

struct WriteResult : net::Status {
  using net::Status::Status;

  std::uint32_t bytes = 0u;
  std::uint32_t header_bytes = 0u;
  std::uint32_t payload_bytes = 0u;
  bool header_written = false;
  bool payload_written = false;
  bool would_block = false;
  bool budget_exhausted = false;

  [[nodiscard]] constexpr bool complete() const noexcept {
    return header_written && payload_written;
  }
};

} // namespace rund::net::frame
