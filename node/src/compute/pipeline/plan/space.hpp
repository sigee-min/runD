#pragma once

#include "../../memory/arena.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::compute::detail {

struct DeviceState;

}

namespace rund::compute::detail::space {

struct Bounds final {
  std::size_t ordinary{};
  std::size_t storage{};
};

[[nodiscard]] inline Bounds bounds(const DeviceState &device) noexcept {
  const std::uint64_t words = memory::arena_bytes(device) / memory::Word;
  const std::size_t storage = words > std::numeric_limits<std::size_t>::max()
                                  ? std::numeric_limits<std::size_t>::max()
                                  : static_cast<std::size_t>(words);
  return Bounds{
      .ordinary =
          std::min(storage, static_cast<std::size_t>(memory::ChunkWords)),
      .storage = storage,
  };
}

[[nodiscard]] inline bool align(const std::size_t value,
                                std::size_t &result) noexcept {
  const std::size_t mask =
      static_cast<std::size_t>(memory::AlignmentWords - 1u);
  if (value > std::numeric_limits<std::size_t>::max() - mask) {
    return false;
  }
  result = (value + mask) & ~mask;
  return true;
}

[[nodiscard]] inline bool align(const std::size_t value,
                                const std::size_t alignment,
                                std::size_t &result) noexcept {
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) {
    return false;
  }
  const std::size_t mask = alignment - 1u;
  if (value > std::numeric_limits<std::size_t>::max() - mask) {
    return false;
  }
  result = (value + mask) & ~mask;
  return true;
}

} // namespace rund::compute::detail::space
