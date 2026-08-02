#pragma once

#include <rund/compute/backend.hpp>

#include <cstddef>

namespace rund::measure::compute {

void PrintPipelineColumns();
void PrintCheckpointColumns();
void PrintRecurrenceColumns();
void PrintNestedRepeatColumns();
void PrintPipelineProfileColumns();
void PrintPreparationMemoryColumns();

[[nodiscard]] bool MeasurePipeline(::rund::compute::Backend backend,
                                   std::size_t count, std::size_t samples);
[[nodiscard]] bool MeasureCheckpoints(::rund::compute::Backend backend,
                                      std::size_t count, std::size_t samples);
[[nodiscard]] bool MeasureRecurrence(::rund::compute::Backend backend,
                                     std::size_t count, std::size_t samples);
[[nodiscard]] bool MeasureNestedRepeat(::rund::compute::Backend backend,
                                       std::size_t samples);
[[nodiscard]] bool MeasurePipelineProfile(::rund::compute::Backend backend,
                                          std::size_t count,
                                          std::size_t samples);
[[nodiscard]] bool MeasurePreparationMemory(::rund::compute::Backend backend,
                                            bool materialize);

} // namespace rund::measure::compute
