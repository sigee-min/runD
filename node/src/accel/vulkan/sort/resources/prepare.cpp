#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../../domain.hpp"
#include "../../kernel/pipeline/template.hpp"
#include "../local/api.hpp"

#include <memory>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "prepare/build.hpp"
#endif

rund::AccelCheck PrepareVulkanSort(const rund::AccelDevice &pick,
                                   const rund::kernel::SortDesc &desc,
                                   const rund::kernel::SortPlan &plan,
                                   const rund::kernel::ComputeDomain domain,
                                   const SortBinds &bindings,
                                   std::shared_ptr<void> &resources,
                                   const VulkanKernelImmutablePipelines
                                       *const pipelines) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  SetVulkanLastError(*adapter, "ok");

  return BuildVulkanSortResources(adapter, pick, desc, plan, domain, bindings,
                                  resources, pipelines);
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)bindings;
  (void)resources;
  (void)pipelines;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
