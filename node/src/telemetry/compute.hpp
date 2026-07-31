#pragma once

#include <rund/compute/stats.hpp>
#include <rund/telemetry/event.hpp>

namespace rund::telemetry::detail {

[[nodiscard]] Event ComputeEvent(const compute::Stats &stats,
                                 compute::Code code, Level level) noexcept;
[[nodiscard]] Findings ComputeFindings(const compute::Stats &stats) noexcept;

} // namespace rund::telemetry::detail
