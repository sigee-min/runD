#include "batch.hpp"

#include "../../resident/access.hpp"
#include "find.hpp"

#include <cstddef>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

void LookupVulkanResidentBatch(const rund::AccelDevice &pick,
                               VulkanResidentReq *const reqs,
                               const std::size_t count,
                               const char *const missing_reason) {
  const bool pick_valid = VulkanPickOwnsAdapter(pick);
  auto *const adapter =
      pick_valid ? static_cast<VulkanAdapter *>(pick.backend.context) : nullptr;
  for (std::size_t index = 0u; index < count; ++index) {
    VulkanResidentReq &req = reqs[index];
    if (req.out == nullptr) {
      continue;
    }
    const bool handle_valid = req.handle != nullptr && *req.handle != nullptr;
    *req.out = RejectResident<VulkanResidentBufferResult>(
        pick_valid && handle_valid ? missing_reason
                                   : "accel_buffer_backend_unavailable");
  }
  if (adapter == nullptr) {
    return;
  }
  VulkanResidentState &resident = VulkanResidents(*adapter);
  std::lock_guard lock{resident.mutex};
  for (std::size_t index = 0u; index < count; ++index) {
    VulkanResidentReq &req = reqs[index];
    if (req.ref == nullptr || req.handle == nullptr || req.out == nullptr ||
        *req.handle == nullptr) {
      continue;
    }
    *req.out = ResolveVulkanResidentBuffer(resident, *req.ref, *req.handle,
                                           missing_reason);
    // Preserve the validated requested window.  Vulkan descriptors can then
    // express contiguous subviews with offset/range and keep the binding
    // zero-copy.
    if (req.out->check.ok) {
      req.out->ref = *req.ref;
    }
  }
}

#endif

} // namespace rund::node::accel::detail
