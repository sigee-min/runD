#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace rund::node::replay_detail::artifact {

[[nodiscard]] inline bool
append(void *const state, const std::span<const std::byte> bytes) noexcept {
  try {
    auto &target = *static_cast<std::vector<std::byte> *>(state);
    target.insert(target.end(), bytes.begin(), bytes.end());
    return true;
  } catch (...) {
    return false;
  }
}

} // namespace rund::node::replay_detail::artifact
