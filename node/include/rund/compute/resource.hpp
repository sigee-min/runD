#pragma once

#include <cstdint>
#include <limits>

namespace rund::compute::resource {

inline constexpr std::uint32_t NoNode =
    std::numeric_limits<std::uint32_t>::max();

enum class AccessMode : unsigned char {
  Read,
  Write,
};

} // namespace rund::compute::resource
