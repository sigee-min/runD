#pragma once

#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

[[nodiscard]] constexpr std::uint64_t
CeilGroups(const std::uint64_t items,
           const std::uint64_t capacity) noexcept {
  return items == 0u || capacity == 0u
             ? 0u
             : 1u + (items - 1u) / capacity;
}

[[nodiscard]] constexpr std::uint64_t
ScanDispatches(const std::uint64_t passes, const std::uint64_t blocks,
               const std::uint64_t limit) noexcept {
  const std::uint64_t chunks = CeilGroups(blocks, limit);
  if ((passes != 1u && passes != 2u) || chunks == 0u) {
    return 0u;
  }
  if (passes == 1u) {
    return chunks;
  }
  constexpr std::uint64_t maximum =
      std::numeric_limits<std::uint64_t>::max();
  return chunks <= (maximum - 1u) / 2u ? 2u * chunks + 1u : 0u;
}

[[nodiscard]] constexpr std::uint64_t
SortDispatches(const std::uint64_t passes, const std::uint64_t blocks,
               const std::uint64_t limit) noexcept {
  const std::uint64_t chunks = CeilGroups(blocks, limit);
  if (passes == 0u || chunks == 0u) {
    return 0u;
  }
  constexpr std::uint64_t maximum =
      std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t fixed = blocks == 1u ? 1u : 2u;
  const std::uint64_t budget = (maximum - 1u) / passes;
  if (budget < fixed || chunks > (budget - fixed) / 2u) {
    return 0u;
  }
  return 1u + passes * (2u * chunks + fixed);
}

static_assert(CeilGroups(65'536u, 65'535u) == 2u);
static_assert(ScanDispatches(2u, 65'536u, 65'535u) == 5u);
static_assert(SortDispatches(8u, 65'536u, 65'535u) == 49u);
static_assert(ScanDispatches(2u, std::numeric_limits<std::uint64_t>::max(),
                            1u) == 0u);

} // namespace rund::node::accel::detail
