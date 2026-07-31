#pragma once

#include <accel/device.hpp>

#include "model.hpp"

#include <cstddef>
#include <memory>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
struct VulkanResidentReq {
  const rund::kernel::ResidentBufferRef *ref = nullptr;
  const std::shared_ptr<void> *handle = nullptr;
  VulkanResidentBufferResult *out = nullptr;
};

void LookupVulkanResidentBatch(const rund::AccelDevice &pick,
                               VulkanResidentReq *reqs, std::size_t count,
                               const char *missing_reason);

template <std::size_t N>
inline void LookupVulkanResidentBatch(const rund::AccelDevice &pick,
                                      VulkanResidentReq (&reqs)[N],
                                      const char *const missing_reason) {
  LookupVulkanResidentBatch(pick, reqs, N, missing_reason);
}
#endif

} // namespace rund::node::accel::detail
