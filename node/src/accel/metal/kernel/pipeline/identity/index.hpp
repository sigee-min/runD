#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>

namespace rund::node::accel::detail {

// Cold Metal finalization preserves first-command order in its retained
// vectors. This one-shot index is membership-only: one allocation is reused
// for pipeline-state and resource deduplication, then discarded before the
// warm submission is published.
struct MetalPointerIdentityIndexLayout final {
  std::uint64_t slot_count{};
  std::uint64_t byte_count{};
  bool ok{};
};

[[nodiscard]] inline MetalPointerIdentityIndexLayout
PlanMetalPointerIdentityIndex(const std::uint64_t entry_capacity) noexcept {
  if (entry_capacity == 0u) {
    return MetalPointerIdentityIndexLayout{.ok = true};
  }
  if (entry_capacity > std::numeric_limits<std::uint64_t>::max() / 2u) {
    return {};
  }
  const std::uint64_t minimum_slots = entry_capacity * 2u;
  std::uint64_t slots = 1u;
  while (slots < minimum_slots) {
    if (slots > std::numeric_limits<std::uint64_t>::max() / 2u) {
      return {};
    }
    slots *= 2u;
  }
  if (slots > std::numeric_limits<std::size_t>::max() ||
      slots >
          std::numeric_limits<std::uint64_t>::max() / sizeof(const void *) ||
      slots * sizeof(const void *) > std::numeric_limits<std::size_t>::max()) {
    return {};
  }
  return MetalPointerIdentityIndexLayout{
      .slot_count = slots,
      .byte_count = slots * sizeof(const void *),
      .ok = true,
  };
}

class MetalPointerIdentityIndex final {
public:
  explicit MetalPointerIdentityIndex(
      const std::uint64_t entry_capacity) noexcept
      : layout_(PlanMetalPointerIdentityIndex(entry_capacity)) {
    if (!layout_.ok || layout_.slot_count == 0u) {
      return;
    }
    slots_.reset(
        new (std::nothrow)
            const void *[static_cast<std::size_t>(layout_.slot_count)]);
    if (slots_ != nullptr) {
      clear();
    }
  }

  [[nodiscard]] bool ready() const noexcept {
    return layout_.ok && (layout_.slot_count == 0u || slots_ != nullptr);
  }

  [[nodiscard]] const MetalPointerIdentityIndexLayout &layout() const noexcept {
    return layout_;
  }

  void clear() noexcept {
    if (slots_ != nullptr) {
      std::fill_n(slots_.get(), static_cast<std::size_t>(layout_.slot_count),
                  nullptr);
    }
    size_ = 0u;
  }

  // Returns false only when the index is unavailable or full. `inserted`
  // distinguishes a new identity from an existing identity without exposing
  // or iterating hash-table order.
  [[nodiscard]] bool insert(const void *const value, bool &inserted) noexcept {
    inserted = false;
    if (value == nullptr) {
      return true;
    }
    if (!ready() || size_ >= layout_.slot_count) {
      return false;
    }
    const std::size_t mask = static_cast<std::size_t>(layout_.slot_count - 1u);
    std::size_t slot = std::hash<const void *>{}(value)&mask;
    for (std::uint64_t probe = 0u; probe < layout_.slot_count; ++probe) {
      const void *const present = slots_[slot];
      if (present == value) {
        return true;
      }
      if (present == nullptr) {
        slots_[slot] = value;
        ++size_;
        inserted = true;
        return true;
      }
      slot = (slot + 1u) & mask;
    }
    return false;
  }

private:
  MetalPointerIdentityIndexLayout layout_{};
  std::unique_ptr<const void *[]> slots_{};
  std::uint64_t size_{};
};

} // namespace rund::node::accel::detail
