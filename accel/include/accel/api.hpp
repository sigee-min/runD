#pragma once

#include <cstdint>

namespace rund {

enum class AccelApi : std::uint8_t {
  Auto = 0u,
  Cpu = 1u,
  Metal = 2u,
  Vulkan = 3u,
  Fake = 4u,
};

}  // namespace rund
