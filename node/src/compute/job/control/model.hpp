#pragma once

#include "../state.hpp"

#include "../../host.hpp"

#include <rund/compute/abi/job.hpp>
#include <rund/compute/abi/observe.hpp>
#include <span>

namespace rund::compute::detail {

enum class JobBindings : unsigned char {
  ReadOnly,
  Writable,
};

[[nodiscard]] Status prepare_cpu_view_transfers(JobState &state);
[[nodiscard]] Status
validate_host_inputs(const ProgramState &program,
                     std::span<const HostView> inputs) noexcept;
[[nodiscard]] Status
validate_bound_buffers(const std::shared_ptr<ProgramState> &program,
                       std::span<const std::shared_ptr<BufferState>> inputs,
                       std::span<const std::shared_ptr<BufferState>> outputs,
                       bool check_poison = true) noexcept;
[[nodiscard]] Status prepare_job_state(const std::shared_ptr<JobState> &state,
                                       JobBindings mode);
[[nodiscard]] Result<std::shared_ptr<JobState>>
finish_prepare(std::shared_ptr<JobState> state, Status status) noexcept;
[[nodiscard]] Result<std::shared_ptr<JobState>>
make_job_values(const std::shared_ptr<ProgramState> &program,
                std::span<const HostView> host_inputs, JobBindings bindings);
[[nodiscard]] Result<RunState>
empty_run(const std::shared_ptr<JobState> &state);
[[nodiscard]] Result<std::shared_ptr<JobState>>
bind_job_validated(const std::shared_ptr<ProgramState> &program,
                   std::span<const std::shared_ptr<BufferState>> inputs,
                   std::span<const std::shared_ptr<BufferState>> outputs);
[[nodiscard]] Result<std::shared_ptr<JobState>>
bind_job(const std::shared_ptr<ProgramState> &program,
         std::span<const std::shared_ptr<BufferState>> inputs,
         std::span<const std::shared_ptr<BufferState>> outputs);
[[nodiscard]] Status refresh_host(const std::shared_ptr<JobState> &state,
                                  std::span<const HostView> inputs) noexcept;
[[nodiscard]] bool
same_buffers(const std::shared_ptr<JobState> &state,
             std::span<const std::shared_ptr<BufferState>> inputs,
             std::span<const std::shared_ptr<BufferState>> outputs) noexcept;
[[nodiscard]] Result<std::size_t>
job_read_size_impl(const std::shared_ptr<JobState> &state,
                   std::size_t logical_output, Type expected_type);
[[nodiscard]] Status job_read_data_impl(const std::shared_ptr<JobState> &state,
                                        std::size_t logical_output,
                                        Type expected_type, void *data,
                                        std::size_t bytes,
                                        std::size_t logical_count,
                                        bool destination_zeroed);

} // namespace rund::compute::detail
