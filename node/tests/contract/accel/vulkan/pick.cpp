#include <accel/api.hpp>
#include <accel/device.hpp>

#include "local.hpp"
#include <node/accel/pick.hpp>

#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/lowering/vulkan/shape.hpp>

#include <string_view>

namespace node_accel_contract {

[[nodiscard]] bool VulkanPickReportsBackendInfo(const rund::AccelDevice &pick) {
  if (!pick.check.ok || pick.backend_info.device_name.empty()) {
    return false;
  }
  return pick.backend_info.driver_name != "MoltenVK" ||
         !pick.backend_info.driver_info.empty();
}

[[nodiscard]] bool VulkanPickHasDiscoveryCaps(const rund::AccelDevice &pick) {
  return pick.check.ok && pick.api == rund::AccelApi::Vulkan && pick.caps.ok &&
         pick.caps.api == rund::kernel::ComputeApi::Vulkan &&
         std::string_view{pick.caps.reason} == "ok" &&
         pick.caps.device_bytes > 0u &&
         pick.caps.staging_bytes >= vulkan::kOneMiB &&
         pick.caps.max_window_tiles >=
             rund::kernel::compute_lowering_detail::kVulkanMapWidth &&
         pick.caps.subgroup_width >= 1u && static_cast<bool>(pick.backend) &&
         pick.owner != nullptr && VulkanPickReportsBackendInfo(pick);
}

[[nodiscard]] bool
RequiredVulkanPickIsPlatformAware(const rund::AccelDevice &pick) {
  if (pick.check.ok) {
    return VulkanPickHasDiscoveryCaps(pick);
  }
  return vulkan::FailureReasonIsPrecise(pick);
}

[[nodiscard]] bool VulkanDirectBackendLastErrorIsPrecise() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(vulkan::RequiredPolicy());
  if (!pick.check.ok) {
    return vulkan::FailureReasonIsPrecise(pick);
  }
  rund::kernel::ComputePlan invalid_plan{};
  rund::kernel::LoweringArtifact artifact{};
  rund::kernel::BindingSet bindings{};
  const rund::kernel::ComputeDispatchWindow window{};
  const bool accepted = pick.backend.execute(pick.backend.context, invalid_plan,
                                             artifact, &window, 1u, bindings);
  if (accepted || pick.backend.last_error == nullptr) {
    return false;
  }
  return std::string_view{pick.backend.last_error(pick.backend.context)} ==
         "compute_plan_invalid";
}

} // namespace node_accel_contract
