#pragma once

#include "../prepared.hpp"
#include "../../job/state.hpp"

namespace rund::compute::detail {

[[nodiscard]] Status prepare_cpu_run(JobState &job);
[[nodiscard]] Status prepare_cpu_run(
    JobState &job, std::shared_ptr<CpuGraphStorage> storage,
    const CpuRunRoutePlan &plan,
    std::shared_ptr<CpuPreparedArena> prepared_arena,
    const CpuRunRouteSlice &route_slice);
[[nodiscard]] Status refresh_cpu_map_bindings(JobState &job) noexcept;

[[nodiscard]] Result<CpuGraphStoragePlan>
plan_cpu_graph_storage(const std::shared_ptr<ProgramState> &program) noexcept;

[[nodiscard]] Result<std::shared_ptr<CpuGraphStorage>>
make_cpu_graph_storage(const std::shared_ptr<ProgramState> &program,
                       const CpuGraphStoragePlan &plan,
                       std::shared_ptr<CpuPreparedArena> prepared_arena);

[[nodiscard]] Result<CpuRunRoutePlan>
plan_cpu_run_route(const std::shared_ptr<ProgramState> &program) noexcept;

[[nodiscard]] Status materialize_cpu_run(
    CpuRun &run, const std::shared_ptr<ProgramState> &program,
    std::span<const std::shared_ptr<BufferState>> workspace = {});
[[nodiscard]] Status materialize_cpu_run(
    CpuRun &run, const std::shared_ptr<ProgramState> &program,
    std::span<const std::shared_ptr<BufferState>> workspace,
    std::shared_ptr<CpuGraphStorage> storage, const CpuRunRoutePlan &plan,
    std::shared_ptr<CpuPreparedArena> prepared_arena,
    const CpuRunRouteSlice &route_slice);

[[nodiscard]] Status execute_cpu_primitive(JobState &job, std::size_t step);
[[nodiscard]] Status reset_cpu(JobState &job, std::size_t step) noexcept;

[[nodiscard]] Result<RunState> run_cpu_job(JobState &job);
[[nodiscard]] Status run_cpu_pipeline_job(JobState &job);

} // namespace rund::compute::detail
