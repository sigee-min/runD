#pragma once

#include "state.hpp"

#include <vector>

namespace rund::compute::detail {

[[nodiscard]] Result<RunState>
empty_job_run(const std::shared_ptr<JobState> &state);
[[nodiscard]] Status run_pipeline_job(const std::shared_ptr<JobState> &state);
[[nodiscard]] Status
gather_cpu_pipeline_views(const std::shared_ptr<JobState> &state) noexcept;
[[nodiscard]] Status
publish_cpu_pipeline_views(const std::shared_ptr<JobState> &state) noexcept;
[[nodiscard]] bool valid_job(const std::shared_ptr<JobState> &state) noexcept;
[[nodiscard]] Status start_queued_job(const std::shared_ptr<JobState> &state);
[[nodiscard]] Result<std::shared_ptr<JobState>>
bind_job_buffers(const std::shared_ptr<ProgramState> &program,
                 std::span<const std::shared_ptr<BufferState>> inputs,
                 std::span<const std::shared_ptr<BufferState>> outputs);
// Pipeline Phase 1 has already validated every static binding invariant and
// Phase 3 has validated poison once in canonical resource order.  This owner
// transfer prepares the private Job without repeating either public scan.
[[nodiscard]] Result<std::shared_ptr<JobState>>
prepare_pipeline_job_buffers(const std::shared_ptr<ProgramState> &program,
                             std::vector<std::shared_ptr<BufferState>> inputs,
                             std::vector<std::shared_ptr<BufferState>> outputs,
                             std::vector<JobBufferView> input_views = {},
                             std::vector<JobBufferView> output_views = {},
                             node::accel::detail::KernelViewLayout views = {},
                             std::shared_ptr<JobWorkspace> workspace = {});

} // namespace rund::compute::detail
