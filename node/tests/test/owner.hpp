#pragma once

#include <memory>

namespace rund::node::test {

[[nodiscard]] inline bool SameOwner(const std::shared_ptr<void> &lhs,
                                    const std::shared_ptr<void> &rhs) noexcept {
  if (lhs == nullptr || rhs == nullptr) {
    return false;
  }
  const std::owner_less<void> less{};
  return !less(lhs, rhs) && !less(rhs, lhs);
}

} // namespace rund::node::test
