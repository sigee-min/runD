#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../reduce/shape.hpp"
#include "../../reduce/vulkan.hpp"
#include "../collective/execute.hpp"
#include "../status.hpp"
#include "encode/pass.hpp"
#include "local.hpp"
#include "resources/buffers.hpp"
#include "resources/lookup.hpp"
#include "resources/passes.hpp"

#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void DestroyVulkanReduceEncodeResources(void *const raw) {
  auto *const resources = static_cast<VulkanReduceEncodeResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  VulkanAdapter *const adapter = resources->adapter;
  if (adapter != nullptr) {
    ReleaseVulkanBuffer(*adapter, resources->params);
    ReleaseVulkanBuffer(*adapter, resources->partial);
    ReleaseVulkanStatus(*adapter, resources->status);
  }
  delete resources;
}
#endif

rund::AccelCheck PrepareVulkanReduce(const rund::AccelDevice &pick,
                                     const rund::kernel::ReduceDesc &desc,
                                     const rund::kernel::ReducePlan &plan,
                                     const rund::kernel::ComputeDomain domain,
                                     const ReduceBinds &bindings,
                                     std::shared_ptr<void> &resources) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  SetVulkanLastError(*adapter, "ok");
  if (!ReduceShapeOk(desc, plan, bindings)) {
    SetVulkanLastError(*adapter, "compute_reduce_invalid");
    return rund::AccelCheck{false, "compute_reduce_invalid"};
  }
  if (!VulkanReduceIndexRangeOk(plan)) {
    SetVulkanLastError(*adapter, "compute_reduce_index_range_invalid");
    return rund::AccelCheck{false, "compute_reduce_index_range_invalid"};
  }
  const ReduceBufferLookup lookup = LookupReduceBuffers(pick, bindings);
  if (!ReduceLookupOk(lookup)) {
    const char *const reason = ReduceLookupReason(lookup);
    SetVulkanLastError(*adapter, reason);
    return rund::AccelCheck{false, reason};
  }

  auto *const raw = new VulkanReduceEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyVulkanReduceEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  const auto requested = [](const VulkanResidentBufferResult &buffer) {
    return VulkanStorageBinding{
        buffer.device_buffer, static_cast<VkDeviceSize>(buffer.ref.offset_bytes),
        static_cast<VkDeviceSize>(buffer.ref.count * buffer.ref.element_bytes)};
  };
  raw->input = requested(lookup.input);
  raw->output = requested(lookup.output);
  raw->logical_count = requested(bindings.logical_count_handle == nullptr
                                     ? lookup.input
                                     : lookup.logical_count);
  raw->pipeline = AcquireReducePipeline(*adapter, desc, domain);
  if (raw->pipeline == nullptr ||
      !CreateVulkanReduceScratchBuffers(*adapter, *raw, plan) ||
      !CreateVulkanReducePassResources(*adapter, *raw)) {
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

rund::AccelCheck EncodeVulkanReduce(VulkanAdapter &adapter,
                                    const std::shared_ptr<void> &resources,
                                    void *const command_buffer_raw) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanReduceEncodeState state{};
  const rund::AccelCheck loaded = LoadVulkanReduceEncodeState(
      adapter, resources, command_buffer_raw, state);
  if (!loaded.ok) {
    return loaded;
  }
  if (!ResetVulkanStatus(state.command, state.reduce->status,
                         state.reduce->plan.status_bytes)) {
    return rund::AccelCheck{false, "compute_reduce_invalid"};
  }
  EncodeVulkanReducePipeline(*state.reduce, state.command);
  if (state.reduce->plan.op == rund::kernel::ReduceOp::Sum ||
      state.reduce->plan.op == rund::kernel::ReduceOp::CountNonzero) {
    const auto groups =
        static_cast<std::uint32_t>(state.reduce->plan.first_pass_group_count);
    EncodeVulkanReduceGroups(*state.reduce, state.command, 0u, groups);
    if (state.reduce->plan.pass_count == 2u) {
      EncodeVulkanReducePartialBarrier(*state.reduce, state.command);
      EncodeVulkanReduceGroups(*state.reduce, state.command, 1u, 1u);
    }
    const std::array<const VulkanBuffer *, 1u> outputs{
        state.reduce->output.buffer};
    if (!FinishVulkanStatus(state.command, state.reduce->status, outputs)) {
      return rund::AccelCheck{false, "compute_reduce_invalid"};
    }
    return rund::AccelCheck{true, "ok"};
  }
  rund::kernel::u64 current = state.reduce->plan.element_count;
  for (rund::kernel::u64 pass = 0u; pass < state.reduce->plan.pass_count;
       ++pass) {
    const rund::kernel::u64 next =
        EncodeVulkanReducePass(*state.reduce, state.command, current, pass);
    if (next == 1u) {
      break;
    }
    EncodeVulkanReducePartialBarrier(*state.reduce, state.command);
    current = next;
  }
  const std::array<const VulkanBuffer *, 1u> outputs{
      state.reduce->output.buffer};
  if (!FinishVulkanStatus(state.command, state.reduce->status, outputs)) {
    return rund::AccelCheck{false, "compute_reduce_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_buffer_raw;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck ExecuteVulkanReduce(const rund::AccelDevice &pick,
                                     const rund::kernel::ReduceDesc &desc,
                                     const rund::kernel::ReducePlan &plan,
                                     const rund::kernel::ComputeDomain domain,
                                     const ReduceBinds &bindings) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return ExecuteVulkanDomainCollective(pick, desc, plan, domain, bindings,
                                       PrepareVulkanReduce, EncodeVulkanReduce,
                                       FinishVulkanReduce);
#else
  (void)domain;
  return RejectVulkanCollectiveExecute(pick, desc, plan, bindings);
#endif
}
} // namespace rund::node::accel::detail
