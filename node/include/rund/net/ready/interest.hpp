#pragma once

#include <cstdint>

namespace rund::net::ready {

enum class Interest : std::uint8_t {
  Readable,
  Writable,
  ReadWrite,
};

} // namespace rund::net::ready
