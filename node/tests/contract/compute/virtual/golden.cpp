#include "golden.hpp"

#include "model.hpp"

#include <bit>
#include <cstddef>

namespace rund_node_test_virtual {

std::int32_t SeedValue(const std::size_t index) noexcept {
  return static_cast<std::int32_t>(index * 11u) - 23;
}

std::int32_t FusedValue(const std::int32_t value,
                        const std::size_t index) noexcept {
  return (value + 5) * 3 - static_cast<std::int32_t>(index % 7u);
}

void SeedInput(const std::span<std::int32_t> output) noexcept {
  for (std::size_t index = 0u; index < output.size(); ++index) {
    output[index] = SeedValue(index);
  }
}

bool GoldenMatches(const std::span<const std::int32_t> values) noexcept {
  if (values.size() != LogicalElements) {
    return false;
  }
  for (std::size_t index = 0u; index < values.size(); ++index) {
    if (values[index] != FusedValue(SeedValue(index), index)) {
      return false;
    }
  }
  return true;
}

std::uint64_t HashValues(const std::span<const std::int32_t> values) noexcept {
  constexpr std::uint64_t offset = 14695981039346656037ull;
  constexpr std::uint64_t prime = 1099511628211ull;
  std::uint64_t hash = offset;
  for (const std::int32_t value : values) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    for (std::uint32_t byte = 0u; byte < 4u; ++byte) {
      hash ^= (bits >> (byte * 8u)) & 0xffu;
      hash *= prime;
    }
  }
  return hash;
}

} // namespace rund_node_test_virtual
