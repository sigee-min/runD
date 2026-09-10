#pragma once

#include "../model.hpp"
#include "graph_residency/internal.hpp"

namespace rund::measure::compute {

namespace virtual_graph_residency {
struct Result;
[[nodiscard]] bool Run(::rund::compute::Device &device,
                       const ::rund::compute::DeviceInfo &info, Result &result);
[[nodiscard]] bool ReportEnvironment(Backend backend,
                                     const ::rund::compute::Device &device,
                                     const ::rund::compute::DeviceInfo &info);
void Report(const Result &result);
} // namespace virtual_graph_residency

void PrintVirtualGraphResidencyColumns();

} // namespace rund::measure::compute
