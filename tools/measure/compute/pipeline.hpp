#pragma once

#include <rund/compute/backend.hpp>

#include <cstddef>

namespace rund::measure::compute {

void PrintPipelineColumns();
void PrintRecurrenceColumns();
void PrintPipelineProfileColumns();

[[nodiscard]] bool MeasurePipeline(::rund::compute::Backend backend,
                                   std::size_t count, std::size_t samples);
[[nodiscard]] bool MeasureRecurrence(::rund::compute::Backend backend,
                                     std::size_t count, std::size_t samples);
[[nodiscard]] bool MeasurePipelineProfile(::rund::compute::Backend backend,
                                          std::size_t count,
                                          std::size_t samples);

} // namespace rund::measure::compute
