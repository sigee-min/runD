#pragma once

#include <rund/compute/abi/model.hpp>
#include <rund/compute/stats.hpp>
#include <span>
namespace rund::compute::detail {
[[nodiscard]] Stats job_stats(const std::shared_ptr<JobState> &state) noexcept;
[[nodiscard]] MemoryStats
device_memory(const std::shared_ptr<DeviceState> &state) noexcept;
[[nodiscard]] MemoryStats
job_memory(const std::shared_ptr<JobState> &state) noexcept;
[[nodiscard]] MemoryStats
program_memory(const std::shared_ptr<ProgramState> &state) noexcept;
[[nodiscard]] MemorySnapshot
device_memory_snapshot(const std::shared_ptr<DeviceState> &state,
                       std::span<MemoryEntry> entries) noexcept;
[[nodiscard]] MemorySnapshot
program_memory_snapshot(const std::shared_ptr<ProgramState> &state,
                        std::span<MemoryEntry> entries) noexcept;
[[nodiscard]] const graph::Info &
program_graph_info(const std::shared_ptr<ProgramState> &state) noexcept;
[[nodiscard]] Result<Backend>
program_backend(const std::shared_ptr<ProgramState> &state) noexcept;
[[nodiscard]] std::size_t
program_input_size(const std::shared_ptr<ProgramState> &state,
                   std::size_t index) noexcept;
[[nodiscard]] std::size_t
program_output_size(const std::shared_ptr<ProgramState> &state,
                    std::size_t index) noexcept;
[[nodiscard]] bool
program_input_sizes_match(const std::shared_ptr<ProgramState> &state,
                          std::span<const std::size_t> sizes) noexcept;
[[nodiscard]] MemorySnapshot
job_memory_snapshot(const std::shared_ptr<JobState> &state,
                    std::span<MemoryEntry> entries) noexcept;
[[nodiscard]] WriteStats
job_write_stats(const std::shared_ptr<JobState> &state) noexcept;
[[nodiscard]] Status write_buffer(const std::shared_ptr<BufferState> &buffer,
                                  HostView input, WriteStats &stats) noexcept;
[[nodiscard]] Stats run_stats(const RunState &run) noexcept;
} // namespace rund::compute::detail
