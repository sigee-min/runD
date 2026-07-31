#pragma once

#include <accel/check.hpp>

#include "../../resident/access.hpp"
#include "input.hpp"

#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] inline bool PrepareVulkanResidentBindings(
    VulkanAdapter &adapter, const rund::kernel::ComputePlan &plan,
    const rund::kernel::BindingSet &bindings, VulkanResidentBindings &out) {
  const char *reason = "ok";
  if (!VulkanResidentBindingShapeOk(plan, bindings, reason)) {
    out.reason = reason;
    return false;
  }
  out = VulkanResidentBindings{};
  out.bindings = &bindings;
  VulkanResidentState &resident = VulkanResidents(adapter);
  std::lock_guard lock{resident.mutex};
  return PrepareVulkanResidentOutput(resident, plan, bindings, out) &&
         PrepareVulkanResidentInputs(resident, plan, bindings, out);
}
#endif

} // namespace rund::node::accel::detail
