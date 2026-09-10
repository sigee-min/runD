#pragma once

#include "../model.hpp"

#include "../../../cpu/run/state.hpp"

#include <kernel/dispatch/kernel.hpp>

namespace rund::compute::detail::cpu_tile_detail {

void begin_trace_dispatch(CpuRun &) noexcept;
void cancel_trace_dispatch(CpuRun &) noexcept;
void finish_trace_dispatch(CpuRun &, bool timestamped) noexcept;

[[nodiscard]] Status
submit_primitive(JobState &, kernel::WorkerBackend, void *ready_context,
                 void (*ready)(void *context) noexcept) noexcept;
[[nodiscard]] kernel::ComputeTileCallbackResult
collective_tile(const void *, const kernel::ComputeTile &) noexcept;
[[nodiscard]] Status
submit_tiles(kernel::ComputeTileExecutor &, kernel::WorkerBackend,
             const void *context, kernel::ComputeTileCallback,
             void *ready_context,
             void (*ready)(void *context) noexcept) noexcept;
void add_tiles(Stats &, const kernel::ComputeTileRunResult &,
               std::uint64_t dispatches, std::uint64_t tile_size) noexcept;

} // namespace rund::compute::detail::cpu_tile_detail
