#include "context.hpp"
#include "../residency/local.hpp"
#include "../../../../kernel/backend/pipeline/failure.hpp"
#include <limits>
#include <rund/counter.hpp>
namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck PrepareVulkanPipelineResidencyMemory(VulkanPipelinePreparation &preparation) {
  auto &pipeline = preparation.pipeline;
  auto &memory = *preparation.memory;
  const rund::AccelCheck residency_ready =
      PrepareVulkanPipelineResidency(*pipeline);
  if (!residency_ready.ok) {
    return FailVulkanPipeline(pipeline, residency_ready.reason);
  }
  if (pipeline->residency != nullptr) {
    std::uint64_t sliding_storage_bytes =
        ::rund::detail::counter::SaturatingAdd(
            static_cast<std::uint64_t>(
                pipeline->residency->sliding.descriptor.allocated_bytes),
            static_cast<std::uint64_t>(pipeline->residency->sliding
                                           .original_arguments
                                           .allocated_bytes));
    sliding_storage_bytes = ::rund::detail::counter::SaturatingAdd(
        sliding_storage_bytes,
        static_cast<std::uint64_t>(
            pipeline->residency->sliding.argument_owners.allocated_bytes));
    const std::uint64_t residency_bytes =
        ::rund::detail::counter::SaturatingAdd(
            static_cast<std::uint64_t>(
                pipeline->residency->arguments.allocated_bytes),
            sliding_storage_bytes);
    if (residency_bytes == std::numeric_limits<std::uint64_t>::max()) {
      return FailVulkanPipeline(pipeline, "compute_pipeline_capacity");
    }
    accumulate_memory(memory.staging,
                      PreparedMemory{.current = residency_bytes,
                                     .peak = residency_bytes,
                                     .cumulative = residency_bytes,
                                     .budget = residency_bytes});
  }

  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
