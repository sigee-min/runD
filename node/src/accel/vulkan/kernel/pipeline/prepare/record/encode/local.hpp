#pragma once

#include "../../../../../command/capture.hpp"
#include "../../../../../command/timestamp.hpp"
#include "../../record.hpp"

namespace rund::node::accel::detail::vulkan_record_detail {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)

// Borrowed invocation context; the pipeline owns every native resource.
struct Encoding final {
  VulkanPipeline &pipeline;
  VulkanPipelineRecordRecipe &recipe;
  VkCommandBuffer recording;
  VulkanDispatchCapture &capture;
  PreparedPipelineFailureContext *failure;
  VulkanTimestampCapture *timestamps;
  const VulkanPipelineRecordSlice *slice;
};

template <typename Encode>
auto Trace(VulkanTimestampCapture *timestamps, Encode &&encode) {
  if (timestamps == nullptr)
    return encode();
  VulkanTimestampScope scope{*timestamps};
  return encode();
}

struct StepEvidence final {
  std::uint32_t template_index;
  PreparedProgramStatusSlice status_range;
  PreparedProgramStatusSlice telemetry_range;
  std::uint32_t failed_outer_window;
  std::uint32_t failed_inner_iteration;
  std::uint32_t failed_nested_phase;
  std::uint32_t window_state;
};

[[nodiscard]] rund::AccelCheck EncodeEntry(const Encoding &, std::size_t index,
                                           bool first,
                                           bool &scratch_seen) noexcept;
[[nodiscard]] rund::AccelCheck EncodeEvidence(const Encoding &,
                                              const StepEvidence &,
                                              std::size_t step_index) noexcept;

#endif
} // namespace rund::node::accel::detail::vulkan_record_detail
