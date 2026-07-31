#pragma once

#include "../adapter/api.hpp"
#include "../buffer/resident/batch.hpp"
#include "../collective/pipeline.hpp"
#include "../descriptor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail {

struct NumericParams {
  rund::kernel::u64 op = 0u;
  rund::kernel::u64 layout = 1u;
  rund::kernel::u64 rows = 0u;
  rund::kernel::u64 cols = 0u;
  rund::kernel::u64 inner = 0u;
  rund::kernel::u64 batch_count = 0u;
  rund::kernel::u64 rhs_cols = 0u;
  rund::kernel::u64 value_count = 0u;
  rund::kernel::u64 vector_count = 0u;
  rund::kernel::u32 mode = 0u;
  rund::kernel::u32 aux = 0u;
  rund::kernel::u32 max_iterations = 0u;
};

static_assert(sizeof(NumericParams) == 88u);

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
enum class VulkanNumericStatusKind : std::uint8_t {
  None,
  Factor,
  Solve,
  Spectrum,
};

struct VulkanNumericPrepared final {
  VulkanAdapter *adapter = nullptr;
  std::array<VulkanResidentBufferResult, 5u> resident{};
  VulkanBuffer params{};
  VulkanBuffer dummy{};
  VulkanBuffer twiddle{};
  VulkanBuffer status_readback{};
  VulkanCollectivePipeline *pipeline = nullptr;
  VkDescriptorSet descriptor = VK_NULL_HANDLE;
  std::array<VulkanStorageBinding, 6u> bindings{};
  std::size_t binding_count = 0u;
  std::array<const VulkanBuffer *, 3u> outputs{};
  std::size_t output_count = 0u;
  VulkanBuffer *status = nullptr;
  VulkanStorageBinding status_binding{};
  rund::kernel::u64 status_count = 0u;
  rund::kernel::u64 dispatches = 0u;
  rund::kernel::u64 transform_count = 0u;
  std::uint32_t groups = 0u;
  VulkanNumericStatusKind status_kind{VulkanNumericStatusKind::None};
  bool pipeline_private{};
};
#endif

} // namespace rund::node::accel::detail
