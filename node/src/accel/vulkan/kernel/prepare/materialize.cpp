#include "../../adapter/error.hpp"
#include "../../adapter/access.hpp"

#include "internal.hpp"

#include <new>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] rund::AccelCheck MaterializeVulkanPrimitivePipelines(
    const rund::AccelDevice &pick, const BoundStep &step,
    const PreparedBackendManifest &manifest,
    std::shared_ptr<const VulkanKernelImmutablePipelines> &out) {
  out.reset();
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr || step.step == nullptr || step.planned == nullptr ||
      step.step->kind() == rund::kernel::NodeKind::Map || !manifest.ok) {
    return {false, "accel_kernel_template_invalid"};
  }
  std::shared_ptr<VulkanKernelImmutablePipelines> pipelines;
  try {
    pipelines = std::make_shared<VulkanKernelImmutablePipelines>();
  } catch (const std::bad_alloc &) {
    return {false, "compute_pipeline_capacity"};
  }
  pipelines->kind = step.step->kind();
  bool complete = false;
  switch (step.step->kind()) {
  case rund::kernel::NodeKind::Scan:
  case rund::kernel::NodeKind::SegmentedScan:
  case rund::kernel::NodeKind::Partition:
    complete = MaterializeVulkanScanPipelines(*adapter, step, *pipelines);
    break;
  case rund::kernel::NodeKind::Stencil:
  case rund::kernel::NodeKind::Window:
    complete = MaterializeVulkanRangePipelines(*adapter, step, *pipelines);
    break;
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
  case rund::kernel::NodeKind::Factor:
  case rund::kernel::NodeKind::Solve:
  case rund::kernel::NodeKind::Spectrum:
    complete = MaterializeVulkanNumericPipelines(*adapter, step, *pipelines);
    break;
  case rund::kernel::NodeKind::SegmentedReduce:
  case rund::kernel::NodeKind::Sort:
  case rund::kernel::NodeKind::Compact:
  case rund::kernel::NodeKind::Gather:
  case rund::kernel::NodeKind::Histogram:
  case rund::kernel::NodeKind::Reduce:
  case rund::kernel::NodeKind::Scatter:
  case rund::kernel::NodeKind::ScatterReduce:
    complete = MaterializeVulkanCollectivePipelines(*adapter, step, *pipelines);
    break;
  case rund::kernel::NodeKind::Map:
    break;
  }
  pipelines->capture_direct_dispatch_count =
      manifest.capture_direct_dispatch_count;
  pipelines->capture_indirect_dispatch_count =
      manifest.capture_indirect_dispatch_count;
  if (!complete || !pipelines->ready(step.step->kind(), manifest)) {
    return {false, VulkanLastError(adapter)};
  }
  out = std::move(pipelines);
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
