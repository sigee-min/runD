#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../stencil/shape.hpp"
#include "../collective/execute.hpp"
#include "encode/barrier.hpp"
#include "local.hpp"
#include "resources/buffers.hpp"
#include "resources/lookup.hpp"

#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void DestroyVulkanStencilEncodeResources(void *const raw) {
  auto *const resources = static_cast<VulkanStencilEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  VulkanAdapter *const adapter = resources->adapter;
  if (adapter != nullptr) {
    ReleaseVulkanBuffer(*adapter, resources->params);
  }
  delete resources;
}
#endif

rund::AccelCheck PrepareVulkanStencil(const rund::AccelDevice &pick,
                                      const rund::kernel::StencilDesc &desc,
                                      const rund::kernel::StencilPlan &plan,
                                      const rund::kernel::ComputeDomain domain,
                                      const StencilBinds &bindings,
                                      std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  SetVulkanLastError(*adapter, "ok");
  if (!StencilShapeOk(desc, plan, bindings)) {
    SetVulkanLastError(*adapter, "compute_stencil_invalid");
    return rund::AccelCheck{false, "compute_stencil_invalid"};
  }
  const StencilBufferLookup lookup = LookupStencilBuffers(pick, bindings);
  if (!StencilLookupOk(lookup)) {
    const char *const reason = StencilLookupReason(lookup);
    SetVulkanLastError(*adapter, reason);
    return rund::AccelCheck{false, reason};
  }

  auto *const raw = new VulkanStencilEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyVulkanStencilEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  raw->input = lookup.input.device_buffer;
  raw->output = lookup.output.device_buffer;
  raw->input_binding =
      VulkanStorageBindingFor(lookup.input.device_buffer, lookup.input.ref);
  raw->output_binding =
      VulkanStorageBindingFor(lookup.output.device_buffer, lookup.output.ref);
  raw->pipeline = AcquireStencilPipeline(*adapter, desc, domain);
  const StencilParams params_value{plan.element_count, plan.radius};
  if (raw->input_binding.buffer == nullptr ||
      raw->output_binding.buffer == nullptr || raw->pipeline == nullptr ||
      !CreateVulkanStencilBuffers(*adapter, *raw, params_value) ||
      !CreateVulkanStencilDescriptorSet(*adapter, *raw)) {
    return rund::AccelCheck{false, VulkanLastError(adapter)};
  }
  resources = std::move(owned);
  return rund::AccelCheck{true, "ok"};
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)bindings;
  (void)resources;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck EncodeVulkanStencil(VulkanAdapter &adapter,
                                     const std::shared_ptr<void> &resources,
                                     void *const command_buffer_raw) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanStencilEncodeState state{};
  const rund::AccelCheck loaded = LoadVulkanStencilEncodeState(
      adapter, resources, command_buffer_raw, state);
  if (!loaded.ok) {
    return loaded;
  }
  EncodeVulkanStencilDispatch(*state.stencil, state.command, state.workgroups);
  EncodeVulkanStencilFinishBarrier(*state.stencil, state.command);
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_buffer_raw;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck ExecuteVulkanStencil(const rund::AccelDevice &pick,
                                      const rund::kernel::StencilDesc &desc,
                                      const rund::kernel::StencilPlan &plan,
                                      const rund::kernel::ComputeDomain domain,
                                      const StencilBinds &bindings) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return ExecuteVulkanDomainCollective(
      pick, desc, plan, domain, bindings, PrepareVulkanStencil,
      EncodeVulkanStencil, FinishVulkanStencil);
#else
  (void)domain;
  return RejectVulkanCollectiveExecute(pick, desc, plan, bindings);
#endif
}

} // namespace rund::node::accel::detail
