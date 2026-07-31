#pragma once

#include "state.hpp"

#include <limits>

namespace rund::node {

[[nodiscard]] constexpr ReactorHandle
ReactorHandleFromPublic(const int value) noexcept {
  return value < 0
             ? kInvalidReactorHandle
             : static_cast<ReactorHandle>(static_cast<unsigned int>(value));
}

[[nodiscard]] constexpr int
ReactorHandleForPublic(const ReactorHandle handle) noexcept {
  return handle == kInvalidReactorHandle ||
                 handle >
                     static_cast<ReactorHandle>(std::numeric_limits<int>::max())
             ? -1
             : static_cast<int>(handle);
}

} // namespace rund::node
