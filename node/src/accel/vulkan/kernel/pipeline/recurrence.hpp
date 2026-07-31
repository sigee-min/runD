#pragma once

#include "state.hpp"

#include "../../../kernel/recurrence.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] rund::AccelCheck PrepareVulkanRecurrence(
    std::span<const BackendBatchEntry> entries, const MapRecurrence &recurrence,
    PreparedPipelineStatusLayout &status, VulkanPipeline &pipeline,
    PreparedMemory &staging_memory);
[[nodiscard]] rund::AccelCheck PrepareVulkanTransducers(
    std::span<const BackendBatchEntry> templates,
    std::span<const TileTransducer> transducers, VulkanPipeline &pipeline,
    PreparedMemory &staging_memory);

#endif

} // namespace rund::node::accel::detail
