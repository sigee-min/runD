#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace rund::node::accel::detail {

struct KernelBindingIndices {
  static constexpr std::size_t kInlineIndexCapacity = 4u;

  std::array<std::uint64_t, kInlineIndexCapacity> inline_indices{};
  std::vector<std::uint64_t> overflow_indices{};
  // The vector owns the overflow length; only the inline arm needs a count.
  std::uint8_t inline_count = 0u;
  bool heap = false;
  bool ok = true;

  void reserve(const std::uint64_t expected) {
    if (!ok || expected >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      ok = false;
      return;
    }
    if (expected > kInlineIndexCapacity) {
      if (!heap && inline_count > kInlineIndexCapacity) {
        ok = false;
        return;
      }
      overflow_indices.reserve(static_cast<std::size_t>(expected));
      if (!heap) {
        for (std::uint64_t index = 0u; index < inline_count; ++index) {
          overflow_indices.push_back(
              inline_indices[static_cast<std::size_t>(index)]);
        }
        inline_count = 0u;
        heap = true;
      }
    }
  }

  [[nodiscard]] bool push_back(const std::uint64_t index) {
    if (!ok || size() == std::numeric_limits<std::uint64_t>::max()) {
      ok = false;
      return false;
    }
    if (!heap && inline_count == kInlineIndexCapacity) {
      reserve(static_cast<std::uint64_t>(kInlineIndexCapacity + 1u));
      if (!ok) {
        return false;
      }
    }
    if (!heap && inline_count < kInlineIndexCapacity) {
      inline_indices[static_cast<std::size_t>(inline_count++)] = index;
    } else {
      overflow_indices.push_back(index);
    }
    return true;
  }

  [[nodiscard]] bool valid() const noexcept {
    return ok ? (heap ? inline_count == 0u
                      : overflow_indices.empty() &&
                            inline_count <= kInlineIndexCapacity)
              : false;
  }

  [[nodiscard]] std::uint64_t size() const noexcept {
    return heap ? static_cast<std::uint64_t>(overflow_indices.size())
                : inline_count;
  }

  [[nodiscard]] std::uint64_t operator[](
      const std::size_t index) const noexcept {
    return heap ? overflow_indices[index] : inline_indices[index];
  }
};

static_assert(sizeof(KernelBindingIndices) <= 64u);

}  // namespace rund::node::accel::detail
