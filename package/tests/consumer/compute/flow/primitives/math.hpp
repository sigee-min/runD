#pragma once

#include "model.hpp"

#include <rund/compute/math.hpp>

namespace package_compute {

inline constexpr auto zero = rund::compute::Fixed<1, 31>::zero();
inline constexpr auto half = rund::compute::Fixed<1, 31>::from_raw(1 << 30);
inline constexpr auto quarter = rund::compute::Fixed<1, 31>::from_raw(1 << 29);
inline constexpr std::array<rund::compute::Fixed<1, 31>, 4> left{half, zero,
                                                                 zero, half};
inline constexpr std::array<rund::compute::Fixed<1, 31>, 4> right{half, half,
                                                                  half, half};
inline constexpr std::array<rund::compute::Fixed<1, 31>, 4> transform_real{
    half, zero, zero, zero};
inline constexpr std::array<rund::compute::Fixed<1, 31>, 4> transform_imag{
    zero, zero, zero, zero};

} // namespace package_compute
