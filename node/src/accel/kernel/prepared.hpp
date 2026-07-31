#pragma once

#include <accel/check.hpp>
#include <accel/context/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/value.hpp>

#include "backend/run.hpp"
#include "memory.hpp"
#include "preparation.hpp"
#include "profile.hpp"
#include "scratch.hpp"
#include "status.hpp"

#include <node/accel/context.hpp>

#include <cstdint>
#include <memory>
#include <span>

namespace rund::node::accel::detail {

struct PreparedKernelRun {
  std::shared_ptr<void> owner{};
  bool ok = false;
  const char *reason = "accel_kernel_run_invalid";
  std::uint32_t failed_node = NoNode;
};

struct PreparedKernelPipeline {
  std::shared_ptr<void> owner{};
  bool ok = false;
  const char *reason = "accel_kernel_pipeline_invalid";
};

struct PreparedBatchEvidence final {
  rund::AccelEvidence shared{};
  rund::AccelCheck check{};
};

struct PreparedPipelineEvidence final {
  rund::AccelEvidence shared{};
  rund::AccelCheck check{};
  PreparedPipelineControl control{};
  PreparedPipelineProfileEvidence profile{};
  std::uint64_t status_entry_count{};
  std::uint64_t control_byte_count{};
  std::uint64_t control_command_count{};
  std::uint64_t control_ns{};
  std::uint32_t active_step_count{};
  bool submitted{};
  bool control_observed{};
  bool control_valid{};
};

using PreparedKernelCompletion = void (*)(void *,
                                          const rund::AccelEvidence &) noexcept;
using PreparedPipelineCompletion =
    void (*)(void *, PreparedPipelineEvidence &&) noexcept;
using PreparedBatchStart = void (*)(void *) noexcept;

[[nodiscard]] PreparedKernelRun
PrepareKernelRun(const rund::AccelContext &context,
                 const rund::AccelKernel &kernel, const rund::AccelRun &run,
                 KernelPreparationMode mode = KernelPreparationMode::Standalone,
                 const KernelViewLayout *views = nullptr,
                 const RunBinds *view_binds = nullptr,
                 const KernelScratchLayout *scratch = nullptr);

[[nodiscard]] rund::AccelEvidence
RunPreparedKernel(const rund::AccelContext &context,
                  const PreparedKernelRun &prepared);

[[nodiscard]] PreparedBatchEvidence
RunPreparedKernelBatch(const rund::AccelContext &context,
                       std::span<const PreparedKernelRun *const> prepared,
                       std::span<rund::AccelEvidence> jobs,
                       std::shared_ptr<void> &workspace,
                       PreparedBatchStart start, void *user);

[[nodiscard]] PreparedKernelPipeline
PrepareKernelPipeline(const rund::AccelContext &context,
                      std::span<const PreparedKernelRun *const> prepared,
                      std::span<const std::uint8_t> barriers,
                      std::span<const std::uint32_t> declared_steps,
                      std::span<const BackendRecurrence> recurrences,
                      std::span<const BackendPublish> publications,
                      std::uint32_t declared_step_count,
                      std::uint32_t generation_stride, bool profile_steps);

[[nodiscard]] PreparedPipelineEvidence
RunPreparedKernelPipeline(const rund::AccelContext &context,
                          const PreparedKernelPipeline &prepared);

[[nodiscard]] rund::AccelCheck
SeedPreparedKernelPipelineGeneration(const PreparedKernelPipeline &prepared,
                                     std::uint32_t generation) noexcept;

[[nodiscard]] rund::AccelCheck SubmitPreparedKernelPipeline(
    const rund::AccelContext &context, const PreparedKernelPipeline &prepared,
    std::shared_ptr<void> lifetime, PreparedPipelineCompletion completion,
    void *user) noexcept;

[[nodiscard]] rund::AccelCheck
SubmitPreparedKernel(const rund::AccelContext &context,
                     const PreparedKernelRun &prepared,
                     std::shared_ptr<void> lifetime,
                     PreparedKernelCompletion completion, void *user) noexcept;

[[nodiscard]] PreparedMemory
ReadPreparedKernelMemory(const PreparedKernelRun &prepared) noexcept;

[[nodiscard]] PreparedPipelineMemory ReadPreparedKernelPipelineMemory(
    const PreparedKernelPipeline &prepared) noexcept;

[[nodiscard]] PreparedPipelineStatusLayout ReadPreparedKernelPipelineStatus(
    const PreparedKernelPipeline &prepared) noexcept;

} // namespace rund::node::accel::detail
