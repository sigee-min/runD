#pragma once

#include "../../adapter/buffer.hpp"
#include "../pool.hpp"

#include <accel/check.hpp>
#include <kernel/program/compute/binding/model.hpp>

#include "../../../resident/model.hpp"

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanAdapter;

struct VulkanResidentBuffer : ResidentEntry {
  std::weak_ptr<void> storage{};
  VulkanBuffer *device_buffer = nullptr;
};

struct VulkanResidentBufferResult {
  rund::AccelCheck check{};
  rund::kernel::ResidentBufferRef ref{};
  std::shared_ptr<void> handle{};
  std::shared_ptr<void> storage{};
  VulkanBuffer *device_buffer = nullptr;
  std::uint64_t storage_bytes = 0u;
  bool storage_reused = false;
};

#endif

} // namespace rund::node::accel::detail
