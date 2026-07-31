#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../gather/shape.hpp"
#include "../collective/execute.hpp"
#include "../status.hpp"
#include "encode/dispatch.hpp"
#include "local.hpp"
#include "resources/buffers.hpp"
#include "resources/lookup.hpp"

#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void DestroyVulkanGatherEncodeResources(void *const raw) {
  auto *const resources = static_cast<VulkanGatherEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  VulkanAdapter *const adapter = resources->adapter;
  if (adapter != nullptr) {
    ReleaseVulkanBuffer(*adapter, resources->params);
    ReleaseVulkanBuffer(*adapter, resources->indirect);
    ReleaseVulkanStatus(*adapter, resources->status);
  }
  delete resources;
}
#endif

rund::AccelCheck PrepareVulkanGather(const rund::AccelDevice &pick,
                                     const rund::kernel::GatherDesc &desc,
                                     const rund::kernel::GatherPlan &plan,
                                     const GatherBinds &bindings,
                                     std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  SetVulkanLastError(*adapter, "ok");
  if (!GatherShapeOk(desc, plan, bindings)) {
    SetVulkanLastError(*adapter, "compute_gather_invalid");
    return rund::AccelCheck{false, "compute_gather_invalid"};
  }
  GatherBufferLookup lookup = LookupGatherBuffers(pick, bindings);
  const bool has_logical_count = bindings.logical_count_handle != nullptr;
  if (!GatherLookupOk(lookup, has_logical_count)) {
    const char *const reason = GatherLookupReason(lookup, has_logical_count);
    SetVulkanLastError(*adapter, reason);
    return rund::AccelCheck{false, reason};
  }

  auto *const raw = new VulkanGatherEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyVulkanGatherEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  raw->values = std::move(lookup.values);
  raw->indices = std::move(lookup.indices);
  raw->logical_count = std::move(lookup.logical_count);
  raw->output = std::move(lookup.output);
  raw->control_pipeline = AcquireGatherPipeline(*adapter, desc, true);
  raw->gather_pipeline = AcquireGatherPipeline(*adapter, desc, false);
  const GatherParams params_value{
      plan.element_count, plan.source_count,
      static_cast<rund::kernel::u32>(plan.count_source), 0u};
  if (raw->control_pipeline == nullptr || raw->gather_pipeline == nullptr ||
      !CreateVulkanGatherBuffers(*adapter, *raw, params_value)) {
    return rund::AccelCheck{false, VulkanLastError(adapter)};
  }
  if (raw->values.device_buffer == nullptr ||
      raw->indices.device_buffer == nullptr ||
      raw->output.device_buffer == nullptr ||
      (plan.count_source != rund::kernel::ComputeCountSource::Descriptor &&
       raw->logical_count.device_buffer == nullptr) ||
      !CreateVulkanGatherDescriptorSet(*adapter, *raw)) {
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
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck EncodeVulkanGather(VulkanAdapter &adapter,
                                    const std::shared_ptr<void> &resources,
                                    void *const command_buffer_raw) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanGatherEncodeState state{};
  const rund::AccelCheck loaded = LoadVulkanGatherEncodeState(
      adapter, resources, command_buffer_raw, state);
  if (!loaded.ok) {
    return loaded;
  }
  if (!ResetVulkanStatus(state.command, state.gather->status,
                         state.gather->plan.status_bytes)) {
    return rund::AccelCheck{false, "compute_gather_invalid"};
  }
  EncodeVulkanGatherDispatch(*state.gather, state.command);
  const std::array<const VulkanBuffer *, 1u> outputs{
      state.gather->output.device_buffer};
  if (!FinishVulkanStatus(state.command, state.gather->status, outputs)) {
    return rund::AccelCheck{false, "compute_gather_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_buffer_raw;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck ExecuteVulkanGather(const rund::AccelDevice &pick,
                                     const rund::kernel::GatherDesc &desc,
                                     const rund::kernel::GatherPlan &plan,
                                     const GatherBinds &bindings) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return ExecuteVulkanCollective(pick, desc, plan, bindings,
                                 PrepareVulkanGather, EncodeVulkanGather,
                                 FinishVulkanGather);
#else
  return RejectVulkanCollectiveExecute(pick, desc, plan, bindings);
#endif
}

} // namespace rund::node::accel::detail
