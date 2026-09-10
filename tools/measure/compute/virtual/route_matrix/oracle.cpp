#include "oracle/local.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::measure::compute::route_matrix::oracle {

using oracle_detail::window_value;

std::int32_t seed_value(const std::uint64_t index) noexcept {
  return static_cast<std::int32_t>((index * 17u + 11u) % 63u) - 31;
}

std::uint64_t hash_values(const std::span<const std::int32_t> values) noexcept {
  constexpr std::uint64_t offset = 1'469'598'103'934'665'603ull;
  constexpr std::uint64_t prime = 1'099'511'628'211ull;
  std::uint64_t hash = offset;
  for (const std::byte value : std::as_bytes(values)) {
    hash ^= std::to_integer<std::uint8_t>(value);
    hash *= prime;
  }
  return hash == 0u ? 1u : hash;
}

void fill_expected(const std::span<std::int32_t> values,
                   const CaseSpec &spec) noexcept {
  for (std::uint64_t index = 0u; index < values.size(); ++index) {
    values[static_cast<std::size_t>(index)] =
        spec.family == "spatial_window"
            ? window_value(index, values.size(), spec.window)
            : static_cast<std::int32_t>((seed_value(index) + 5) * 3);
  }
}

bool output_ok(const std::span<const std::int32_t> values, const CaseSpec &spec,
               const std::uint64_t expected_hash) noexcept {
  if (spec.q == 0u ||
      spec.q > std::numeric_limits<std::uint64_t>::max() / PageElements) {
    return false;
  }
  const std::uint64_t count = spec.q * PageElements - 1u;
  if (values.size() != count) {
    return false;
  }
  for (std::uint64_t index = 0u; index < count; ++index) {
    const std::int32_t expected =
        spec.family == "spatial_window"
            ? window_value(index, count, spec.window)
            : static_cast<std::int32_t>((seed_value(index) + 5) * 3);
    if (values[static_cast<std::size_t>(index)] != expected) {
      return false;
    }
  }
  return hash_values(values) == expected_hash;
}

} // namespace rund::measure::compute::route_matrix::oracle
