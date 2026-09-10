#include "local.hpp"

namespace rund::measure::compute::virtual_window {

std::int32_t seed_value(const std::size_t index) noexcept {
  return static_cast<std::int32_t>(index % 4'093u) - 2'046;
}

void seed(const std::span<std::int32_t> values) noexcept {
  for (std::size_t index = 0u; index < values.size(); ++index) {
    values[index] = seed_value(index);
  }
}

std::int32_t expected_value(const std::size_t index) noexcept {
  return (seed_value(index) + 5) * 3;
}

} // namespace rund::measure::compute::virtual_window
