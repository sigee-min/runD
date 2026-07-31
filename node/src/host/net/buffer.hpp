#pragma once

#include <cstddef>

namespace rund::net {

[[nodiscard]] inline bool InvalidBuffer(const void *const data,
                                        const std::size_t size) noexcept {
  return data == nullptr && size != 0u;
}

} // namespace rund::net
