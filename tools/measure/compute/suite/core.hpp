#pragma once

#include "api.hpp"

#include <cstddef>
#include <string_view>

namespace rund::measure::compute {

// Each timed resident invocation follows one completed invocation of the same
// prepared job. This is a steady prepared execution, not an idle-to-active
// submission latency sample.
inline constexpr std::size_t kPrimeRuns = 1u;

#if defined(RUND_COMPUTE_FOCUS)
[[nodiscard]] bool ParseBackend(std::string_view name,
                                Backend &backend) noexcept;
#endif

void PrintCsv(std::string_view text);
[[nodiscard]] bool ReportEnvironment(Backend backend,
                                     const ::rund::compute::Device &device);
[[nodiscard]] bool ReportEnvironment(Backend backend);

} // namespace rund::measure::compute
