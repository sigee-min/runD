#pragma once

#include <cstdint>

namespace rund::net::flow {

struct State {
  std::uint64_t inflight_bytes = 0u;
  std::uint64_t total_bytes = 0u;
  std::uint64_t rejected_bytes = 0u;
};

} // namespace rund::net::flow
