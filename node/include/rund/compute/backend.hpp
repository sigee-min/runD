#pragma once

namespace rund::compute {

enum class Backend : unsigned char {
  Cpu,
  Metal,
  Vulkan,
  Unavailable,
};

} // namespace rund::compute
