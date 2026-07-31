#pragma once

#include "../buffer/resident/model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanResidentState final {
  std::mutex mutex{};
  std::unordered_map<std::uint64_t, VulkanResidentBuffer> buffers{};
  std::array<VulkanBuffer, kVulkanPoolCapacity> pool{};
  std::size_t pool_size = 0u;
  std::uint64_t pool_bytes = 0u;
  std::uint64_t next_id = 1u;
};

#endif

} // namespace rund::node::accel::detail
