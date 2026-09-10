#include <accel/device.hpp>

#include "../../adapter/access.hpp"
#include "../../resident/access.hpp"
#include "../create/memory.hpp"
#include "find.hpp"
#include "view.hpp"

#include <cstddef>
#include <limits>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] VulkanResidentBufferResult ResolveHostVisible(
    const rund::AccelDevice &pick,
    const rund::kernel::ResidentBufferRef &requested,
    const std::shared_ptr<void> &handle) noexcept {
  if (!VulkanPickOwnsAdapter(pick) || handle == nullptr ||
      requested.offset_bytes != 0u || requested.bytes == 0u) {
    return {};
  }
  auto *const adapter = static_cast<VulkanAdapter *>(pick.backend.context);
  VulkanResidentState &resident = VulkanResidents(*adapter);
  std::lock_guard lock{resident.mutex};
  VulkanResidentBufferResult resolved = ResolveVulkanResidentBuffer(
      resident, requested, handle, "compute_resident_id_invalid");
  if (!resolved.check.ok || resolved.device_buffer == nullptr ||
      !VulkanHostCoherentBufferReady(*resolved.device_buffer) ||
      requested.bytes > resolved.device_buffer->bytes ||
      requested.bytes > std::numeric_limits<std::size_t>::max()) {
    return {};
  }
  return resolved;
}

} // namespace

BackendHostView ReadVulkanResidentBuffer(
    const rund::AccelDevice &pick,
    const rund::kernel::ResidentBufferRef &requested,
    const std::shared_ptr<void> &handle) noexcept {
  const VulkanResidentBufferResult resolved =
      ResolveHostVisible(pick, requested, handle);
  return !resolved.check.ok
             ? BackendHostView{}
             : BackendHostView{
                   .data = static_cast<const std::byte *>(
                       resolved.device_buffer->mapped),
                   .bytes = requested.bytes,
               };
}

BackendHostWriteView WriteVulkanResidentBuffer(
    const rund::AccelDevice &pick,
    const rund::kernel::ResidentBufferRef &requested,
    const std::shared_ptr<void> &handle) noexcept {
  const VulkanResidentBufferResult resolved =
      ResolveHostVisible(pick, requested, handle);
  return !resolved.check.ok
             ? BackendHostWriteView{}
             : BackendHostWriteView{
                   .data =
                       static_cast<std::byte *>(resolved.device_buffer->mapped),
                   .bytes = requested.bytes,
               };
}
#endif

} // namespace rund::node::accel::detail
