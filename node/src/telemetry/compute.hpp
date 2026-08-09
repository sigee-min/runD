#pragma once

#include <rund/compute/telemetry.hpp>
#include <rund/telemetry/event.hpp>

namespace rund::telemetry::detail {

[[nodiscard]] Event ProjectEvent(const compute::telemetry::Profile &profile,
                                 compute::Code code, Level level) noexcept;

} // namespace rund::telemetry::detail
