#pragma once

#include "../../kernel/backend/run.hpp"
#include "../../kernel/callback.hpp"
#include "../../kernel/memory.hpp"

#include <accel/check.hpp>

#include <memory>

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck RunCpuKernel(const BackendRun &run);
[[nodiscard]] rund::AccelCheck PrepareCpuKernel(const BackendRun &run,
                                                std::shared_ptr<void> &prepared,
                                                PreparedMemory &memory);
[[nodiscard]] rund::AccelCheck SubmitPreparedCpuKernel(
    const BackendRun &run, const std::shared_ptr<void> &prepared,
    KernelCompletion completion, void *user, PreparedMemoryMeter *memory,
    const std::shared_ptr<void> &lifetime) noexcept;

} // namespace rund::node::accel::detail
