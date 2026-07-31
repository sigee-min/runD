#pragma once

#include <cstdint>

namespace rund::net::frame {

struct Header {
  std::uint32_t bytes = 0u;
};

} // namespace rund::net::frame
