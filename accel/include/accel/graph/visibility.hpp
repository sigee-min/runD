#pragma once

#include <cstdint>

namespace rund {

enum class GraphBufferVisibility : std::uint8_t {
  External = 1u,
  Internal = 2u,
};

} // namespace rund
