#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../kernel/backend/run.hpp"
#include "../kernel/callback.hpp"
#include "../kernel/memory.hpp"
#include "../kernel/preparation.hpp"

#include <cstddef>
#include <memory>

namespace rund::node::accel::detail {

struct VulkanAdapter;

[[nodiscard]] rund::AccelCheck PrepareVulkanResources(
    const rund::AccelDevice &pick, const BoundStep *steps,
    std::size_t step_count, std::uint64_t dispatch_count,
    KernelPreparationMode mode, const BoundResets *resets,
    const KernelViewLayout *views, const RunBinds *view_binds,
    const KernelScratchLayout *scratch, const BackendRun *template_probe,
    PreparedKernelTemplateRegistry *templates, std::uint32_t *failed_node,
    std::shared_ptr<void> &prepared, PreparedMemory &memory);
[[nodiscard]] std::uint64_t
VulkanKernelTraffic(const std::shared_ptr<void> &prepared) noexcept;
[[nodiscard]] PreparedMemory
ObserveVulkanPipelineTemplate(const void *prepared) noexcept;
[[nodiscard]] rund::AccelCheck
RunVulkanResources(const rund::AccelDevice &pick,
                   const std::shared_ptr<void> &prepared);
[[nodiscard]] rund::AccelCheck
SubmitVulkanResources(const rund::AccelDevice &pick,
                      const std::shared_ptr<void> &prepared,
                      KernelCompletion completion, void *user) noexcept;
[[nodiscard]] rund::AccelCheck
SeedPreparedVulkanPipelineGeneration(const std::shared_ptr<void> &prepared,
                                     std::uint32_t generation) noexcept;
} // namespace rund::node::accel::detail
