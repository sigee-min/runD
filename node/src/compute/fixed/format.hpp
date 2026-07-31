#pragma once

#include <rund/compute/abi/model.hpp>

#include <kernel/program/compute/model.hpp>

namespace rund::compute::detail {

[[nodiscard]] constexpr kernel::ComputeFixedFormat
kernel_format(const FixedFormat format) noexcept {
  if (format.integer_bits == 0u && format.fraction_bits == 0u) {
    return {};
  }
  return kernel::ComputeFixedFormat{
      .integer_bits = format.integer_bits,
      .fraction_bits = format.fraction_bits,
      .rounding = static_cast<kernel::ComputeRounding>(
          static_cast<unsigned char>(format.rounding) + 1u),
      .overflow = static_cast<kernel::ComputeOverflow>(
          static_cast<unsigned char>(format.overflow) + 1u),
      .approximation = static_cast<kernel::ComputeApproximation>(
          static_cast<unsigned char>(format.approximation) + 1u),
  };
}

} // namespace rund::compute::detail
