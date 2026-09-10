#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rund::compute::detail {

inline constexpr std::size_t MaxOutputs = 16u;
inline constexpr std::size_t MaxMapInputs = 16u;
struct ValueIds final {
  [[nodiscard]] constexpr bool empty() const noexcept { return count == 0u; }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return count; }
  [[nodiscard]] constexpr std::uint32_t front() const noexcept {
    return values.front();
  }
  [[nodiscard]] constexpr std::uint32_t
  operator[](const std::size_t index) const noexcept {
    return values[index];
  }
  [[nodiscard]] constexpr const std::uint32_t *begin() const noexcept {
    return values.data();
  }
  [[nodiscard]] constexpr const std::uint32_t *end() const noexcept {
    return values.data() + count;
  }
  constexpr void push_back(const std::uint32_t value) noexcept {
    values[count++] = value;
  }

private:
  std::array<std::uint32_t, MaxOutputs> values{};
  std::uint32_t count{};
};
static_assert(std::is_trivially_copyable_v<ValueIds>);
static_assert(sizeof(ValueIds) == sizeof(std::uint32_t) * (MaxOutputs + 1u));
struct BoundedIds final {
  std::uint32_t values{};
  std::uint32_t count{};
};
struct ComplexIds final {
  std::uint32_t real{};
  std::uint32_t imag{};
};
struct SolveIds final {
  std::uint32_t values{};
  std::uint32_t status{};
};
struct FactorIds final {
  std::uint32_t packed{};
  std::uint32_t pivots{};
  std::uint32_t status{};
};
struct SpectrumIds final {
  std::uint32_t values{};
  std::uint32_t vectors{};
  std::uint32_t status{};
};

} // namespace rund::compute::detail
