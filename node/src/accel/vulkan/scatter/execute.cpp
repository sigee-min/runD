#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../scatter/shape.hpp"
#include "../collective/execute.hpp"
#include "../status.hpp"
#include "../kernel/pipeline/template.hpp"
#include "encode/dispatch.hpp"
#include "local.hpp"
#include "resources/buffers.hpp"
#include "resources/lookup.hpp"

#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void DestroyVulkanScatterEncodeResources(void *const raw) {
  auto *const resources = static_cast<VulkanScatterEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  VulkanAdapter *const adapter = resources->adapter;
  if (adapter != nullptr) {
    ReleaseVulkanBuffer(*adapter, resources->params);
    ReleaseVulkanStatus(*adapter, resources->status);
  }
  delete resources;
}
#endif

rund::AccelCheck PrepareVulkanScatter(const rund::AccelDevice &pick,
                                      const rund::kernel::ScatterDesc &desc,
                                      const rund::kernel::ScatterPlan &plan,
                                      const ScatterBinds &bindings,
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
  if (!ScatterShapeOk(desc, plan, bindings)) {
    SetVulkanLastError(*adapter, "compute_scatter_invalid");
    return rund::AccelCheck{false, "compute_scatter_invalid"};
  }
  const ScatterBufferLookup lookup = LookupScatterBuffers(pick, bindings);
  if (!ScatterLookupOk(lookup)) {
    const char *const reason = ScatterLookupReason(lookup);
    SetVulkanLastError(*adapter, reason);
    return rund::AccelCheck{false, reason};
  }

  auto *const raw = new VulkanScatterEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyVulkanScatterEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  raw->values = lookup.values.device_buffer;
  raw->indices = lookup.indices.device_buffer;
  raw->output = lookup.output.device_buffer;
  raw->values_binding =
      VulkanStorageBindingFor(lookup.values.device_buffer, lookup.values.ref);
  raw->indices_binding =
      VulkanStorageBindingFor(lookup.indices.device_buffer, lookup.indices.ref);
  raw->output_binding =
      VulkanStorageBindingFor(lookup.output.device_buffer, lookup.output.ref);
  raw->pipeline =
      pipelines == nullptr
          ? AcquireScatterPipeline(*adapter, desc)
          : pipelines->borrow(rund::kernel::NodeKind::Scatter, 1u, 0u,
                              kScatterDescriptorCount, 1u);
  const ScatterParams params_value{plan.element_count, plan.output_count};
  if (raw->values_binding.buffer == nullptr ||
      raw->indices_binding.buffer == nullptr ||
      raw->output_binding.buffer == nullptr || raw->pipeline == nullptr ||
      !CreateVulkanScatterBuffers(*adapter, *raw, params_value) ||
      !CreateVulkanScatterDescriptorSet(*adapter, *raw)) {
    return rund::AccelCheck{false, VulkanLastError(adapter)};
  }
  resources = std::move(owned);
  return rund::AccelCheck{true, "ok"};
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)bindings;
  (void)resources;
  (void)pipelines;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck EncodeVulkanScatter(VulkanAdapter &adapter,
                                     const std::shared_ptr<void> &resources,
                                     void *const command_buffer_raw) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanScatterEncodeState state{};
  const rund::AccelCheck loaded = LoadVulkanScatterEncodeState(
      adapter, resources, command_buffer_raw, state);
  if (!loaded.ok) {
    return loaded;
  }
  if (!ResetVulkanStatus(state.command, state.scatter->status,
                         state.scatter->plan.status_bytes,
                         ~std::uint32_t{0u})) {
    return rund::AccelCheck{false, "compute_scatter_invalid"};
  }
  EncodeVulkanScatterDispatch(*state.scatter, state.command, state.workgroups);
  const std::array<const VulkanBuffer *, 1u> outputs{state.scatter->output};
  if (!FinishVulkanStatus(state.command, state.scatter->status, outputs)) {
    return rund::AccelCheck{false, "compute_scatter_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_buffer_raw;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck ExecuteVulkanScatter(const rund::AccelDevice &pick,
                                      const rund::kernel::ScatterDesc &desc,
                                      const rund::kernel::ScatterPlan &plan,
                                      const ScatterBinds &bindings) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return ExecuteVulkanCollective(pick, desc, plan, bindings,
                                 [](const rund::AccelDevice &device,
                                    const rund::kernel::ScatterDesc &operation,
                                    const rund::kernel::ScatterPlan &prepared,
                                    const ScatterBinds &resident,
                                    std::shared_ptr<void> &resources) {
                                   return PrepareVulkanScatter(
                                       device, operation, prepared, resident,
                                       resources, nullptr);
                                 },
                                 EncodeVulkanScatter,
                                 FinishVulkanScatter);
#else
  return RejectVulkanCollectiveExecute(pick, desc, plan, bindings);
#endif
}

} // namespace rund::node::accel::detail
