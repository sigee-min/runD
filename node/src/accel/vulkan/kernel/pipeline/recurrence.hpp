#pragma once

#include "state.hpp"

#include "../../../kernel/recurrence.hpp"

#include <type_traits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

// One collision-safe immutable recurrence template shared by every equal
// group and by both transactional streams. The transformed artifact is never
// retained: its final source moves into the adapter cache on the miss path.
struct VulkanMapRecurrenceTemplate final {
  VulkanKernelTemplateKind kind{VulkanKernelTemplateKind::MapRecurrence};
  const BackendRun *signature{};
  std::uint64_t group_capacity{};
  std::uint64_t descriptor_set_capacity{};
  std::shared_ptr<const VulkanMapTemplateResources> prepared{};
  std::shared_ptr<VulkanMapDescriptorArena> descriptors{};
  bool history{};
};

static_assert(std::is_standard_layout_v<VulkanMapRecurrenceTemplate>);

[[nodiscard]] rund::AccelCheck PrepareVulkanRecurrence(
    std::span<const BackendBatchEntry> entries, const MapRecurrence &recurrence,
    PreparedKernelTemplateRegistry &registry,
    PreparedPipelineStatusLayout &status, VulkanPipeline &pipeline,
    PreparedMemory &staging_memory);
[[nodiscard]] rund::AccelCheck PrepareVulkanTransducers(
    std::span<const BackendBatchEntry> templates,
    std::span<const TileTransducer> transducers,
    PreparedKernelTemplateRegistry &registry,
    const PreparedPipelineStatusLayout &status, VulkanPipeline &pipeline,
    PreparedMemory &staging_memory);

#endif

} // namespace rund::node::accel::detail
