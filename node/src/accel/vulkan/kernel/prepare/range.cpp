#include "internal.hpp"

#include "../../range/local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool MaterializeVulkanRangePipelines(
    VulkanAdapter &adapter, const BoundStep &step,
    VulkanKernelImmutablePipelines &pipelines) {
  const RangePlan *const range =
      step.step == nullptr ? nullptr : RangePlanFor(step.step->operation);
  const std::optional<RangeExec> execution =
      range == nullptr ? std::nullopt : RangeExec::from(*range);
  bool complete = execution.has_value();
  if (complete) {
    const std::uint32_t descriptor_count = execution->descriptor_count();
    if (step.step->kind() == rund::kernel::NodeKind::Window &&
        range->shape().resident_counted()) {
      complete = pipelines.append_control(
          AcquireVulkanRangeControlPipeline(adapter, *range), 4u, 1u);
    }
    VulkanCollectivePipeline *const data_pipeline =
        complete ? AcquireVulkanRangePipeline(adapter, *execution) : nullptr;
    complete = complete && data_pipeline != nullptr;
    for (std::size_t index = 0u; index < range->stage_count(); ++index) {
      complete =
          complete && pipelines.append(data_pipeline, descriptor_count, 1u);
    }
  }
  return complete;
}

#endif

} // namespace rund::node::accel::detail
