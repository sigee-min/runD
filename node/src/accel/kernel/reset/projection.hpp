#pragma once

#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail::reset {

template <typename Resources, typename Transfer>
[[nodiscard]] bool Find(Resources &resources, const std::uint64_t binding,
                        const Transfer *&replacement) noexcept {
  replacement = nullptr;
  for (std::size_t index = 0u; index < resources.size(); ++index) {
    const auto *const entry = resources.entry(index);
    if (entry == nullptr || entry->view == nullptr ||
        binding >= entry->view->transfer_by_binding.size()) {
      continue;
    }
    const std::uint32_t ordinal =
        entry->view->transfer_by_binding[static_cast<std::size_t>(binding)];
    if (ordinal == 0u) {
      continue;
    }
    if (ordinal > entry->view->transfers.size() ||
        entry->view->transfers[ordinal - 1u].input || replacement != nullptr) {
      replacement = nullptr;
      return false;
    }
    replacement = &entry->view->transfers[ordinal - 1u];
  }
  return true;
}

} // namespace rund::node::accel::detail::reset
