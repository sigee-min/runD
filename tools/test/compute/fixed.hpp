#pragma once

#include <kernel/program/compute/model.hpp>

namespace test {

[[nodiscard]] constexpr rund::kernel::ComputeFixedFormat
FixedFormatForLane(const rund::kernel::ComputeScalar scalar,
                   const rund::kernel::ComputeApproximation approximation =
                       rund::kernel::ComputeApproximation::Exact) noexcept {
  switch (scalar) {
  case rund::kernel::ComputeScalar::Lane32:
    return rund::kernel::ComputeFixedFormat{
        .integer_bits = 1u,
        .fraction_bits = 31u,
        .rounding = rund::kernel::ComputeRounding::NearestEven,
        .overflow = rund::kernel::ComputeOverflow::Saturate,
        .approximation = approximation,
    };
  case rund::kernel::ComputeScalar::Lane64:
    return rund::kernel::ComputeFixedFormat{
        .integer_bits = 1u,
        .fraction_bits = 63u,
        .rounding = rund::kernel::ComputeRounding::NearestEven,
        .overflow = rund::kernel::ComputeOverflow::Saturate,
        .approximation = approximation,
    };
  }
  return {};
}

} // namespace test
