#pragma once

#include <cstdint>

namespace rund::replay {

struct Limits final {
  std::uint64_t max_bytes = 2ull * 1024ull * 1024ull * 1024ull;
  std::uint64_t max_entries = 4ull * 1024ull * 1024ull;
  std::uint64_t max_payload_bytes = 1024ull * 1024ull * 1024ull;
  std::uint64_t max_state_bytes = 64ull * 1024ull * 1024ull;
};

} // namespace rund::replay
