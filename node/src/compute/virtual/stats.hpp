#pragma once

#include <rund/compute/stats.hpp>
#include <rund/compute/status.hpp>

namespace rund::compute::detail {

[[nodiscard]] Status accumulate_virtual_epoch(Stats &total,
                                              const Stats &epoch) noexcept;

} // namespace rund::compute::detail
