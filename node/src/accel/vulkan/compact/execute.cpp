#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../compact/shape.hpp"
#include "../collective/execute.hpp"
#include "../status.hpp"
#include "encode/scatter.hpp"
#include "local.hpp"
#include "resources/buffers.hpp"
#include "resources/lookup.hpp"

#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void DestroyVulkanCompactEncodeResources(void *const raw) {
  auto *const resources = static_cast<VulkanCompactEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  VulkanAdapter *const adapter = resources->adapter;
  if (adapter != nullptr) {
    ReleaseVulkanBuffer(*adapter, resources->params);
    ReleaseVulkanBuffer(*adapter, resources->counts);
    ReleaseVulkanBuffer(*adapter, resources->offsets);
    ReleaseVulkanStatus(*adapter, resources->status);
  }
  delete resources;
}
#endif

rund::AccelCheck PrepareVulkanCompact(const rund::AccelDevice &pick,
                                      const rund::kernel::CompactDesc &desc,
                                      const rund::kernel::CompactPlan &plan,
                                      const CompactBinds &bindings,
                                      std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  SetVulkanLastError(*adapter, "ok");
  if (!CompactShapeOk(desc, plan, bindings)) {
    SetVulkanLastError(*adapter, "compute_compact_invalid");
    return rund::AccelCheck{false, "compute_compact_invalid"};
  }
  const CompactBufferLookup lookup = LookupCompactBuffers(pick, bindings);
  if (!CompactLookupOk(lookup)) {
    const char *const reason = CompactLookupReason(lookup);
    SetVulkanLastError(*adapter, reason);
    return rund::AccelCheck{false, reason};
  }

  auto *const raw = new VulkanCompactEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyVulkanCompactEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  raw->output = lookup.output.device_buffer;
  raw->block_count = static_cast<rund::kernel::u32>(
      (plan.element_count - 1u) / block::VulkanCompact + 1u);
  raw->classify_pipeline =
      AcquireCompactPipeline(*adapter, desc, CompactStage::Classify);
  raw->prefix_pipeline =
      AcquireCompactPipeline(*adapter, desc, CompactStage::Prefix);
  raw->scatter_pipeline =
      AcquireCompactPipeline(*adapter, desc, CompactStage::Scatter);
  const CompactParams params_value{plan.element_count, plan.output_capacity};
  if (raw->classify_pipeline == nullptr || raw->prefix_pipeline == nullptr ||
      raw->scatter_pipeline == nullptr ||
      !CreateVulkanCompactScratchBuffers(*adapter, *raw, params_value) ||
      !CreateVulkanCompactDescriptorSets(*adapter, *raw, lookup.flags,
                                         lookup.output)) {
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

rund::AccelCheck EncodeVulkanCompact(VulkanAdapter &adapter,
                                     const std::shared_ptr<void> &resources,
                                     void *const command_buffer_raw) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanCompactEncodeState state{};
  const rund::AccelCheck loaded = LoadVulkanCompactEncodeState(
      adapter, resources, command_buffer_raw, state);
  if (!loaded.ok) {
    return loaded;
  }
  EncodeVulkanCompactClassify(*state.compact, state.command, state.workgroups);
  EncodeVulkanCompactCountsBarrier(*state.compact, state.command);
  EncodeVulkanCompactPrefix(*state.compact, state.command);
  EncodeVulkanCompactOffsetsBarrier(*state.compact, state.command);
  EncodeVulkanCompactScatter(*state.compact, state.command, state.workgroups);
  const std::array<const VulkanBuffer *, 1u> outputs{state.compact->output};
  if (!FinishVulkanStatus(state.command, state.compact->status, outputs)) {
    return rund::AccelCheck{false, "compute_compact_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_buffer_raw;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck ExecuteVulkanCompact(const rund::AccelDevice &pick,
                                      const rund::kernel::CompactDesc &desc,
                                      const rund::kernel::CompactPlan &plan,
                                      const CompactBinds &bindings) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return ExecuteVulkanCollective(pick, desc, plan, bindings,
                                 PrepareVulkanCompact, EncodeVulkanCompact,
                                 FinishVulkanCompact);
#else
  return RejectVulkanCollectiveExecute(pick, desc, plan, bindings);
#endif
}

} // namespace rund::node::accel::detail
