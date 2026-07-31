#pragma once

#include <cstdint>

namespace rund::node {

struct StopSourceRecord {
  std::uint64_t scheduler_id = 0u;
  std::uint64_t id = 0u;
  std::uint64_t generation = 0u;
  std::uint64_t epoch = 0u;
  bool requested = false;
};

} // namespace rund::node
