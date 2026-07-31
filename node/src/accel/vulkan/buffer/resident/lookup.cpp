#include <accel/device.hpp>

#include "../../adapter/api.hpp"
#include "../../resident/access.hpp"
#include "find.hpp"

#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
VulkanResidentBufferResult
LookupVulkanResidentBuffer(const rund::AccelDevice &pick,
                           const rund::kernel::ResidentBufferRef &ref,
                           const std::shared_ptr<void> &handle) {
  if (!VulkanPickOwnsAdapter(pick) || handle == nullptr) {
    return RejectResident<VulkanResidentBufferResult>(
        "accel_buffer_backend_unavailable");
  }
  auto *const adapter = static_cast<VulkanAdapter *>(pick.backend.context);
  VulkanResidentState &resident = VulkanResidents(*adapter);
  std::lock_guard lock{resident.mutex};
  return ResolveVulkanResidentBuffer(resident, ref, handle,
                                     "compute_resident_id_invalid");
}
#endif

} // namespace rund::node::accel::detail
