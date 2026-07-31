#pragma once

#include <cstdint>

namespace rund::host::random {

struct RunSeed {
  std::uint64_t value = 0xC2B2AE3D27D4EB4Full;
};

} // namespace rund::host::random
