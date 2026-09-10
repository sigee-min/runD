#pragma once

#include "../../local.hpp"

#include <rund/compute/pipeline.hpp>
#include <rund/counter.hpp>

#include "../../../backend.hpp"
#include "../../../device/info.hpp"
#include "../../../job/local.hpp"
#include "../../../stats.hpp"
#include "../../../status.hpp"
#include "../../../terminal.hpp"
#include "../../claim.hpp"
#include "../../run/memory.hpp"
#include "../../run/result.hpp"

namespace rund::compute::detail::async_step_detail {

[[nodiscard]] bool valid_nested_window(std::size_t,
                                       const PipelineWindow &) noexcept;
[[nodiscard]] std::shared_ptr<JobState>
selected_pipeline_job(PipelineState &, const PipelineStep &) noexcept;
void record_pipeline_failure(PipelineState &, std::size_t index,
                             bool outer_known = false, std::size_t outer = 0u,
                             bool inner_known = false,
                             std::size_t inner = 0u) noexcept;
[[nodiscard]] Status
complete_pipeline_step_locked(PipelineState &, std::size_t index, Status,
                              CpuPipelineSchedule *) noexcept;

} // namespace rund::compute::detail::async_step_detail
