#pragma once

#include <rund/compute/stats.hpp>
#include <rund/compute/status.hpp>

namespace rund::compute::detail {

[[nodiscard]] Status accumulate_virtual_wave(Stats &total,
                                             const Stats &wave) noexcept;

} // namespace rund::compute::detail
