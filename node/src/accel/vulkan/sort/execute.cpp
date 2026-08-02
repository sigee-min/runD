#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../collective/execute.hpp"
#include "encode/dispatch.hpp"
#include "encode/pass.hpp"
#include "local/api.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck EncodeVulkanSort(VulkanAdapter &adapter,
                                  const std::shared_ptr<void> &resources,
                                  void *const command_buffer_raw) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanSortEncodeState state{};
  const rund::AccelCheck loaded =
      LoadVulkanSortEncodeState(adapter, resources, command_buffer_raw, state);
  if (!loaded.ok) {
    return loaded;
  }
  EncodeVulkanSortDispatch(*state.sort, state.command);
  for (std::size_t pass = 0u; pass < state.sort->pass_count; ++pass) {
    const rund::AccelCheck encoded =
        EncodeVulkanSortPass(adapter, *state.sort, pass, state.command);
    if (!encoded.ok) {
      return encoded;
    }
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_buffer_raw;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck ExecuteVulkanSort(const rund::AccelDevice &pick,
                                   const rund::kernel::SortDesc &desc,
                                   const rund::kernel::SortPlan &plan,
                                   const rund::kernel::ComputeDomain domain,
                                   const SortBinds &bindings) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return ExecuteVulkanDomainCollective(pick, desc, plan, domain, bindings,
                                       [](const rund::AccelDevice &device,
                                          const rund::kernel::SortDesc &operation,
                                          const rund::kernel::SortPlan &prepared,
                                          const rund::kernel::ComputeDomain active_domain,
                                          const SortBinds &resident,
                                          std::shared_ptr<void> &resources) {
                                         return PrepareVulkanSort(
                                             device, operation, prepared,
                                             active_domain, resident, resources,
                                             nullptr);
                                       },
                                       EncodeVulkanSort,
                                       FinishVulkanSort);
#else
  (void)domain;
  return RejectVulkanCollectiveExecute(pick, desc, plan, bindings);
#endif
}
} // namespace rund::node::accel::detail
