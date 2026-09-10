#pragma once

#include <cstddef>
#include <cstdint>

namespace rund::compute::detail {

struct BoundedInputSchema final {
  std::uint32_t count{};
  std::size_t capacity{};
};
// Canonical resident execution control attached to an otherwise ordinary
// functional Flow step. Values are logical Flow resource ids; no backend
// handle or host callback enters graph identity.
struct FlowControl final {
  std::uint32_t count{};
  std::uint32_t predicate{};
  std::size_t capacity{};
  std::uint64_t predicate_expected{};
  std::uint32_t iteration{};

  [[nodiscard]] constexpr bool empty() const noexcept {
    return count == 0u && predicate == 0u;
  }
};

} // namespace rund::compute::detail
