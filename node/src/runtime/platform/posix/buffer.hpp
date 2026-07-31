#pragma once

#include <cstddef>

namespace rund::node::posix::buffer {

[[nodiscard]] constexpr bool valid(const void *const data,
                                   const std::size_t size) noexcept {
  return data != nullptr || size == 0u;
}

} // namespace rund::node::posix::buffer
