#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

template <typename Entries>
[[nodiscard]] inline auto* ResidentIndexedSlot(
    Entries& entries,
    const std::uint64_t id) noexcept {
  if (id == 0u || id - 1u > std::numeric_limits<std::size_t>::max()) {
    return static_cast<typename Entries::value_type*>(nullptr);
  }
  const std::size_t index = static_cast<std::size_t>(id - 1u);
  return index < entries.size() ? &entries[index] : nullptr;
}

template <typename Entries>
[[nodiscard]] inline auto* ResidentSlot(Entries& entries,
                                        const std::uint64_t id) noexcept {
  auto* const entry = ResidentIndexedSlot(entries, id);
  return entry != nullptr && entry->id == id ? entry : nullptr;
}

template <typename Entries>
[[nodiscard]] inline bool ResidentAppendId(const Entries& entries,
                                           const std::uint64_t id) noexcept {
  return id != 0u && id - 1u == entries.size();
}

}  // namespace rund::node::accel::detail
