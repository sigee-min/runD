#pragma once

#include <cstdint>

namespace rund::net::frame {

struct Limit {
  std::uint32_t max_bytes = 64u * 1024u;
};

struct IoLimit {
  Limit frame{};
  std::uint32_t max_reads = 64u;
  std::uint32_t max_writes = 64u;
};

} // namespace rund::net::frame
