#pragma once

#include "../../local.hpp"
#include "../result.hpp"

#include <cstddef>

namespace rund::compute::detail {

[[nodiscard]] Status seal_cpu_resident(const CpuPipelinePublicationContext &,
                                       std::size_t, const PipelineWindow &,
                                       const PipelineWindowProgress &) noexcept;

[[nodiscard]] Status execute_cpu_pipeline_step(PipelineState &,
                                               PipelineOutcome &,
                                               std::size_t index);

[[nodiscard]] Status run_cpu_ordinary_step(PipelineState &, PipelineOutcome &,
                                           std::size_t index);

[[nodiscard]] bool run_cpu_nested(PipelineState &, PipelineOutcome &,
                                  std::size_t &index);

} // namespace rund::compute::detail
