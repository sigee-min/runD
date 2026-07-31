#pragma once

#include "../callback.hpp"
#include "../memory.hpp"
#include "run.hpp"

#include <accel/check.hpp>

#include <memory>
#include <span>

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck RunFakeKernel(const BackendRun &run);
[[nodiscard]] rund::AccelCheck RunMetalKernel(const BackendRun &run);
[[nodiscard]] rund::AccelCheck RunVulkanKernel(const BackendRun &run);

[[nodiscard]] rund::AccelCheck
PrepareFakeKernel(const BackendRun &run, std::shared_ptr<void> &prepared,
                  PreparedMemory &memory);
[[nodiscard]] rund::AccelCheck SubmitPreparedFakeKernel(
    const BackendRun &run, const std::shared_ptr<void> &prepared,
    KernelCompletion completion, void *user, PreparedMemoryMeter *memory,
    const std::shared_ptr<void> &lifetime) noexcept;

[[nodiscard]] rund::AccelCheck
PrepareMetalKernel(const BackendRun &run, std::shared_ptr<void> &prepared,
                   PreparedMemory &memory);
[[nodiscard]] rund::AccelCheck
PrepareMetalPipelinePrivateKernel(const BackendRun &run,
                                  std::shared_ptr<void> &prepared,
                                  PreparedMemory &memory);
[[nodiscard]] rund::AccelCheck
RunPreparedMetalBatch(std::span<const BackendBatchEntry> entries,
                      std::span<rund::AccelCheck> results,
                      std::shared_ptr<void> &workspace,
                      rund::RuntimeStats &stats);
[[nodiscard]] rund::AccelCheck
PrepareMetalPipeline(std::span<const BackendBatchEntry> templates,
                     std::span<const BackendBatchEntry> entries,
                     std::span<const std::uint8_t> barriers,
                     std::span<const BackendPublish> publications,
                     PreparedPipelineStatusLayout &status, bool profile_steps,
                     std::shared_ptr<void> &prepared,
                     PreparedPipelineMemory &memory);
[[nodiscard]] rund::AccelCheck
SubmitPreparedMetalPipeline(const std::shared_ptr<void> &prepared,
                            KernelCompletion completion, void *user) noexcept;
[[nodiscard]] rund::AccelCheck SubmitPreparedMetalKernel(
    const BackendRun &run, const std::shared_ptr<void> &prepared,
    KernelCompletion completion, void *user, PreparedMemoryMeter *memory,
    const std::shared_ptr<void> &lifetime) noexcept;

[[nodiscard]] rund::AccelCheck
PrepareVulkanKernel(const BackendRun &run, std::shared_ptr<void> &prepared,
                    PreparedMemory &memory);
[[nodiscard]] rund::AccelCheck
PrepareVulkanPipelinePrivateKernel(const BackendRun &run,
                                   std::shared_ptr<void> &prepared,
                                   PreparedMemory &memory);
[[nodiscard]] rund::AccelCheck
RunPreparedVulkanBatch(std::span<const BackendBatchEntry> entries,
                       std::span<rund::AccelCheck> results,
                       std::shared_ptr<void> &workspace,
                       rund::RuntimeStats &stats);
[[nodiscard]] rund::AccelCheck
PrepareVulkanPipeline(std::span<const BackendBatchEntry> templates,
                      std::span<const BackendBatchEntry> entries,
                      std::span<const std::uint8_t> barriers,
                      std::span<const BackendPublish> publications,
                      PreparedPipelineStatusLayout &status, bool profile_steps,
                      std::shared_ptr<void> &prepared,
                      PreparedPipelineMemory &memory);
[[nodiscard]] rund::AccelCheck
SubmitPreparedVulkanPipeline(const std::shared_ptr<void> &prepared,
                             KernelCompletion completion, void *user) noexcept;
[[nodiscard]] rund::AccelCheck SubmitPreparedVulkanKernel(
    const BackendRun &run, const std::shared_ptr<void> &prepared,
    KernelCompletion completion, void *user, PreparedMemoryMeter *memory,
    const std::shared_ptr<void> &lifetime) noexcept;

} // namespace rund::node::accel::detail
