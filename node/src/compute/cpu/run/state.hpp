#pragma once

#include "../../job/state.hpp"

namespace rund::compute::detail {

[[nodiscard]] Status prepare_cpu_run(JobState &job);
[[nodiscard]] Status refresh_cpu_map_bindings(JobState &job) noexcept;

[[nodiscard]] Result<std::unique_ptr<CpuRun>>
make_cpu_run(const std::shared_ptr<ProgramState> &program,
             std::span<const std::shared_ptr<BufferState>> workspace = {});

[[nodiscard]] Status execute_cpu_primitive(JobState &job, std::size_t step);
[[nodiscard]] Status reset_cpu(JobState &job, std::size_t step) noexcept;

[[nodiscard]] Result<RunState> run_cpu_job(JobState &job);
[[nodiscard]] Status run_cpu_pipeline_job(JobState &job);

} // namespace rund::compute::detail
