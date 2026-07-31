#pragma once

#include <kernel/program/compute/matrix/model.hpp>

#include <array>
#include <cstdint>

namespace rund::node::accel::detail {

// Numeric policy is immutable graph identity.  GPU backends consume these four
// values as pipeline specialization constants; they are never runtime data.
struct NumericPolicy final {
  std::uint32_t matrix_arithmetic =
      static_cast<std::uint32_t>(kernel::MatrixArithmetic::Fixed);
  std::uint32_t fraction_bits = 1u;
  std::uint32_t rounding =
      static_cast<std::uint32_t>(kernel::ComputeRounding::TowardZero);
  std::uint32_t overflow =
      static_cast<std::uint32_t>(kernel::ComputeOverflow::Saturate);

  [[nodiscard]] constexpr std::array<std::uint32_t, 4u>
  constants() const noexcept {
    return {matrix_arithmetic, fraction_bits, rounding, overflow};
  }
};

[[nodiscard]] constexpr NumericPolicy
FixedPolicy(const kernel::ComputeFixedFormat format) noexcept {
  return NumericPolicy{
      .matrix_arithmetic =
          static_cast<std::uint32_t>(kernel::MatrixArithmetic::Fixed),
      .fraction_bits = format.fraction_bits,
      .rounding = static_cast<std::uint32_t>(format.rounding),
      .overflow = static_cast<std::uint32_t>(format.overflow),
  };
}

[[nodiscard]] constexpr NumericPolicy
MatrixPolicy(const kernel::MatrixArithmetic arithmetic,
             const kernel::ComputeFixedFormat format) noexcept {
  if (arithmetic == kernel::MatrixArithmetic::Fixed) {
    return FixedPolicy(format);
  }
  // Fixed helpers are unreachable in an integer Matrix specialization.  Keep
  // their dormant shifts well-defined so every compiler may eliminate them.
  return NumericPolicy{
      .matrix_arithmetic = static_cast<std::uint32_t>(arithmetic),
      .fraction_bits = 1u,
      .rounding =
          static_cast<std::uint32_t>(kernel::ComputeRounding::TowardZero),
      .overflow = static_cast<std::uint32_t>(kernel::ComputeOverflow::Wrap),
  };
}

} // namespace rund::node::accel::detail
