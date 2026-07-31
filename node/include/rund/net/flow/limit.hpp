#pragma once

#include <cstdint>

namespace rund::net::flow {

struct Limit {
  std::uint64_t max_inflight_bytes = 64u * 1024u;
  std::uint64_t max_total_bytes = 16u * 1024u * 1024u;
};

} // namespace rund::net::flow
