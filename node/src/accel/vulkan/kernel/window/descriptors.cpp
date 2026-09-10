#include "../../adapter/error.hpp"

#include "internal.hpp"
#include "source.hpp"

#include "../lease.hpp"

#include "../../collective/pipeline.hpp"
#include "../../descriptor.hpp"

#include <array>
#include <limits>
#include <string>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] VulkanCollectivePipeline *
AcquireWindowPipeline(VulkanAdapter &adapter) {
  const rund::kernel::ComputePlan plan{
      .op_hash_hi = 0x7265736964656e74ull,
      .op_hash_lo = 0x7374617465626974ull,
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
  const rund::kernel::LoweringArtifact artifact = VulkanWindowArtifact();
  if (!artifact.ok) {
    return nullptr;
  }
  return AcquireVulkanCollectivePipeline(
      adapter, 8u, sizeof(VulkanWindowParams), plan, artifact);
}

[[nodiscard]] VulkanCollectivePipeline *
AcquireGatePipeline(VulkanAdapter &adapter) {
  const rund::kernel::ComputePlan plan{
      .op_hash_hi = 0x7265736964656e74ull,
      .op_hash_lo = 0x6761746562697431ull,
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
  const rund::kernel::LoweringArtifact artifact = VulkanGateArtifact();
  if (!artifact.ok) {
    return nullptr;
  }
  return AcquireVulkanCollectivePipeline(adapter, 3u, VulkanGateParameterBytes,
                                         plan, artifact);
}

[[nodiscard]] const char *DescriptorFailure(VulkanAdapter &adapter) noexcept {
  const char *const reason = VulkanLastError(&adapter);
  return reason == nullptr || reason[0] == '\0'
             ? "accel_vulkan_descriptor_unavailable"
             : reason;
}

[[nodiscard]] bool SameBinding(const VulkanStorageBinding &left,
                               const VulkanStorageBinding &right) noexcept {
  return left.buffer == right.buffer && left.offset == right.offset &&
         left.range == right.range;
}

} // namespace

rund::AccelCheck
PrepareVulkanWindowDescriptors(VulkanAdapter &adapter,
                               const VulkanBuffer &control,
                               VulkanWindowResources &resources) {
  constexpr std::size_t no_owner = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> descriptor_owners(resources.state_count, no_owner);
  std::size_t unique_state_count = 0u;
  for (std::size_t index = 0u; index < resources.routes.size(); ++index) {
    const VulkanWindowRoute &route = resources.routes[index];
    if (route.params.state >= descriptor_owners.size()) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    std::size_t &owner = descriptor_owners[route.params.state];
    if (owner == no_owner) {
      owner = index;
      ++unique_state_count;
      continue;
    }
    const VulkanWindowRoute &first = resources.routes[owner];
    bool same = SameBinding(route.count_binding, first.count_binding);
    for (std::size_t bank = 0u; same && bank < route.terminal_bindings.size();
         ++bank) {
      same = SameBinding(route.terminal_bindings[bank],
                         first.terminal_bindings[bank]);
    }
    if (!same) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
  }
  if (resources.gate_capacity >
      std::numeric_limits<std::size_t>::max() - unique_state_count) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  resources.descriptor_leases.reserve(
      unique_state_count + static_cast<std::size_t>(resources.gate_capacity));
  resources.pipeline = AcquireWindowPipeline(adapter);
  resources.gate_pipeline =
      resources.gate_capacity == 0u ? nullptr : AcquireGatePipeline(adapter);
  bool ready =
      resources.pipeline != nullptr &&
      (resources.gate_capacity == 0u || resources.gate_pipeline != nullptr) &&
      ReserveVulkanCollectiveDescriptorDemand(adapter, *resources.pipeline, 8u,
                                              unique_state_count) &&
      (resources.gate_capacity == 0u ||
       ReserveVulkanCollectiveDescriptorDemand(
           adapter, *resources.gate_pipeline, 3u, resources.gate_capacity));
  VulkanLeaseScope lease_scope{adapter, resources.descriptor_leases};
  for (std::size_t index = 0u; ready && index < resources.routes.size();
       ++index) {
    VulkanWindowRoute &route = resources.routes[index];
    const std::size_t owner = descriptor_owners[route.params.state];
    if (owner != index) {
      const VulkanWindowRoute &first = resources.routes[owner];
      ready = first.descriptor != VK_NULL_HANDLE;
      route.descriptor = first.descriptor;
      continue;
    }
    ready = AcquireVulkanCollectiveDescriptorSet(adapter, *resources.pipeline,
                                                 8u, route.descriptor);
    if (!ready) {
      break;
    }
    const std::array<VulkanStorageBinding, 8u> bindings{
        route.terminal_bindings[0],
        route.terminal_bindings[1],
        route.terminal_bindings[2],
        route.count_binding,
        VulkanStorageBindingFor(resources.states),
        VulkanStorageBindingFor(resources.arguments),
        VulkanStorageBindingFor(resources.owners),
        VulkanStorageBindingFor(control)};
    ready =
        WriteVulkanStorageDescriptorSet(adapter, route.descriptor, bindings);
  }
  if (!ready) {
    return rund::AccelCheck{false, DescriptorFailure(adapter)};
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
