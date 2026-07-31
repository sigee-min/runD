#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

namespace rund::compute::detail::graph_build_detail {

template <class Key, std::size_t Count> class Index final {
  static constexpr std::size_t Capacity = Count * 2u;
  static_assert(Count != 0u);
  static_assert((Capacity & (Capacity - 1u)) == 0u);

public:
  [[nodiscard]] std::optional<std::size_t> find(const Key key) const noexcept {
    if (key == Key{}) {
      return std::nullopt;
    }
    std::size_t slot = hash(key);
    for (std::size_t probe = 0u; probe < Count; ++probe) {
      const Entry &entry = entries_[slot];
      if (entry.key == Key{}) {
        return std::nullopt;
      }
      if (entry.key == key) {
        return entry.ordinal;
      }
      slot = (slot + 1u) & (Capacity - 1u);
    }
    return std::nullopt;
  }

  [[nodiscard]] std::pair<std::size_t, bool>
  admit(const Key key, const std::size_t ordinal) noexcept {
    std::size_t slot = hash(key);
    for (std::size_t probe = 0u; probe < Capacity; ++probe) {
      Entry &entry = entries_[slot];
      if (entry.key == key) {
        return {entry.ordinal, false};
      }
      if (entry.key == Key{}) {
        entry = Entry{.key = key, .ordinal = ordinal};
        return {ordinal, true};
      }
      slot = (slot + 1u) & (Capacity - 1u);
    }
    return {ordinal, false};
  }

  [[nodiscard]] bool add(const Key key) noexcept {
    return admit(key, 0u).second;
  }

private:
  struct Entry final {
    Key key{};
    std::size_t ordinal{};
  };

  [[nodiscard]] static std::size_t hash(const Key key) noexcept {
    if constexpr (std::is_pointer_v<Key>) {
      return (reinterpret_cast<std::uintptr_t>(key) >> 4u) & (Capacity - 1u);
    } else {
      return (static_cast<std::size_t>(key) * 0x9e3779b1u) & (Capacity - 1u);
    }
  }

  std::array<Entry, Capacity> entries_{};
};

} // namespace rund::compute::detail::graph_build_detail
