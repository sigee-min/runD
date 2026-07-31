#pragma once

#include <accel/check.hpp>

#include "../../../resident/ref.hpp"
#include "../../../resident/result.hpp"
#include "../../../resident/slot.hpp"
#include "../../../resident/validation.hpp"
#include "../../resident/state.hpp"
#include "../local.hpp"
#include <memory>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
struct VulkanResidentMatch {
  VulkanResidentBuffer *entry = nullptr;
  std::shared_ptr<void> owner{};
  const char *reason = nullptr;
};

[[nodiscard]] inline VulkanResidentMatch MatchVulkanResidentBuffer(
    VulkanResidentBuffer &entry, const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle, const bool allow_stride = false) {
  std::shared_ptr<void> owner = entry.owner.lock();
  const char *reason = "ok";
  if (!ResidentRefFits(entry, owner, ref, handle, "compute_resident_id_invalid",
                       reason, allow_stride)) {
    return VulkanResidentMatch{.entry = nullptr, .reason = reason};
  }
  return VulkanResidentMatch{
      .entry = &entry, .owner = std::move(owner), .reason = "ok"};
}

[[nodiscard]] inline VulkanResidentMatch FindVulkanResidentBuffer(
    VulkanResidentState &resident, const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle, const char *const missing_reason,
    const bool allow_stride = false) {
  const auto found = resident.buffers.find(ref.id);
  return found == resident.buffers.end()
             ? VulkanResidentMatch{.entry = nullptr, .reason = missing_reason}
             : MatchVulkanResidentBuffer(found->second, ref, handle,
                                         allow_stride);
}

[[nodiscard]] inline VulkanResidentBufferResult
VulkanResidentResult(VulkanResidentBuffer &entry, std::shared_ptr<void> owner) {
  std::shared_ptr<void> storage = entry.storage.lock();
  if (owner == nullptr || storage == nullptr ||
      entry.device_buffer == nullptr) {
    return RejectResident<VulkanResidentBufferResult>(
        "accel_buffer_unavailable");
  }
  return VulkanResidentBufferResult{
      .check = rund::AccelCheck{true, "ok"},
      .ref = RefFromResident(entry),
      .handle = owner,
      .storage = storage,
      .device_buffer = entry.device_buffer,
  };
}

[[nodiscard]] inline VulkanResidentBufferResult ResolveVulkanResidentBuffer(
    VulkanResidentState &resident, const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle, const char *const missing_reason,
    const bool allow_stride = false) {
  const VulkanResidentMatch match = FindVulkanResidentBuffer(
      resident, ref, handle, missing_reason, allow_stride);
  return match.entry == nullptr
             ? RejectResident<VulkanResidentBufferResult>(match.reason)
             : VulkanResidentResult(*match.entry, std::move(match.owner));
}
#endif
} // namespace rund::node::accel::detail
