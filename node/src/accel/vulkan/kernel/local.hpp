#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../kernel/backend/template_plan.hpp"
#include "../../kernel/reset/model.hpp"
#include "../../kernel/submission.hpp"
#include "../adapter/api.hpp"
#include "../command.hpp"
#include "../command/model.hpp"
#include "../kernel.hpp"
#include "../map/local.hpp"
#include "manifest.hpp"
#include "ops/model.hpp"
#include "pipeline/template.hpp"
#include "view.hpp"
#include <memory>
#include <type_traits>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "local/resources.hpp"

struct VulkanKernelContext {
  VulkanAdapter *adapter = nullptr;
};

[[nodiscard]] rund::AccelCheck
ValidateVulkanKernelContext(const rund::AccelDevice &pick,
                            VulkanKernelContext &out);

[[nodiscard]] rund::AccelCheck
PrepareVulkanStep(const rund::AccelDevice &pick, const BoundStep &step,
                  const VulkanKernelOps &ops, KernelPreparationMode mode,
                  const VulkanKernelImmutablePipelines *pipelines,
                  std::shared_ptr<void> &resources);

[[nodiscard]] rund::AccelCheck PrepareVulkanKernelProgramTemplate(
    const rund::AccelDevice &pick, const BoundStep *steps,
    std::size_t step_count, KernelPreparationMode mode,
    const BackendRun *template_probe, PreparedKernelTemplateRegistry *templates,
    std::uint32_t *failed_node, VulkanKernelResources &resources);

[[nodiscard]] rund::AccelCheck
PrepareVulkanStepViews(const rund::AccelDevice &pick, const BoundStep *steps,
                       std::size_t step_count, KernelPreparationMode mode,
                       const KernelViewLayout *views,
                       const RunBinds *view_binds, std::uint32_t *failed_node,
                       VulkanKernelResources &resources);

[[nodiscard]] rund::AccelCheck
PrepareVulkanSteps(const rund::AccelDevice &pick, const BoundStep *steps,
                   std::size_t step_count, KernelPreparationMode mode,
                   std::uint32_t *failed_node,
                   VulkanKernelResources &resources);

[[nodiscard]] rund::AccelCheck
RecordVulkanKernel(VulkanAdapter &adapter, VulkanKernelResources &resources);

[[nodiscard]] rund::AccelCheck EncodeVulkanStep(VulkanAdapter &adapter,
                                                VulkanKernelEntry &entry,
                                                VkCommandBuffer command);

[[nodiscard]] rund::AccelCheck
EncodeVulkanKernelSteps(VulkanAdapter &adapter,
                        VulkanKernelResources &resources,
                        VkCommandBuffer command);
[[nodiscard]] rund::AccelCheck
EncodeVulkanResets(VulkanKernelResources &resources, std::size_t step,
                   VkCommandBuffer command);

[[nodiscard]] rund::AccelCheck
ExecuteVulkanKernel(VulkanAdapter &adapter,
                    const VulkanKernelResources &resources);

void DestroyVulkanKernelCommand(VulkanAdapter &adapter,
                                VulkanKernelResources &resources) noexcept;

[[nodiscard]] rund::AccelCheck
FinishVulkanStep(VulkanAdapter &adapter, const VulkanKernelOps &ops,
                 const std::shared_ptr<void> &resources);

[[nodiscard]] rund::AccelCheck
FinishVulkanSteps(VulkanAdapter &adapter, VulkanKernelResources &resources,
                  rund::RuntimeStats *stats = nullptr);

#endif
} // namespace rund::node::accel::detail
