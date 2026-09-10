#include "golden.hpp"

#include "model.hpp"

#include <bit>
#include <cstddef>

namespace rund_node_test_virtual::product {

std::int32_t SeedValue(const std::size_t index) noexcept {
  return static_cast<std::int32_t>(index * 11u) - 23;
}

std::int32_t ProductValue(const std::int32_t value) noexcept {
  return (value + 5) * 3;
}

void SeedInput(const std::span<std::int32_t> values) noexcept {
  for (std::size_t index = 0u; index < values.size(); ++index) {
    values[index] = SeedValue(index);
  }
}

bool GoldenMatches(const std::span<const std::int32_t> values) noexcept {
  return values.size() == LogicalElements && GoldenPageMatches(values, 0u);
}

bool GoldenPageMatches(const std::span<const std::int32_t> values,
                       const std::size_t offset) noexcept {
  if (offset > LogicalElements || values.size() > LogicalElements - offset) {
    return false;
  }
  for (std::size_t index = 0u; index < values.size(); ++index) {
    if (values[index] != ProductValue(SeedValue(offset + index))) {
      return false;
    }
  }
  return true;
}

std::uint64_t HashValues(const std::span<const std::int32_t> values) noexcept {
  // VirtualPipeline's terminal output identity uses the repository FNV
  // authority, whose historical offset is intentionally not the IETF value.
  constexpr std::uint64_t offset = 1469598103934665603ull;
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

} // namespace rund_node_test_virtual::product
