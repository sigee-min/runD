#pragma once

#include "state.hpp"

#include "../status.hpp"

namespace rund::compute::detail {

[[nodiscard]] bool merge_cpu_execution_storage_plan(
    CpuExecutionStoragePlan &target,
    const CpuExecutionStoragePlan &source) noexcept;
[[nodiscard]] bool
cpu_execution_storage_required(const CpuExecutionStoragePlan &plan) noexcept;
[[nodiscard]] CpuStorageBytes
cpu_execution_storage_payload(const CpuExecutionStoragePlan &plan) noexcept;

[[nodiscard]] bool append_cpu_run_route_slice(CpuPreparedArenaPlan &arena,
                                              const CpuRunRoutePlan &route,
                                              CpuRunRouteSlice &slice) noexcept;
[[nodiscard]] bool
append_cpu_job_binding_slice(CpuPreparedArenaPlan &arena,
                             const CpuJobBindingCounts &counts,
                             CpuJobBindingSlice &slice) noexcept;
[[nodiscard]] bool
append_cpu_workspace_slice(CpuPreparedArenaPlan &arena,
                           std::size_t buffer_count,
                           CpuWorkspaceSlice &slice) noexcept;
[[nodiscard]] bool
seal_cpu_prepared_arena_plan(CpuPreparedArenaPlan &arena,
                             std::uint64_t page_bytes) noexcept;
[[nodiscard]] CpuStorageBytes
cpu_prepared_arena_payload(const CpuPreparedArenaPlan &plan) noexcept;
[[nodiscard]] Result<std::shared_ptr<CpuPreparedArena>>
make_cpu_prepared_arena(const CpuPreparedArenaPlan &plan);

} // namespace rund::compute::detail
