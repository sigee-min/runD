#include "internal.hpp"

#include "../../../kernel.hpp"
#include "../../manifest.hpp"

#include "../route.hpp"
#include "capture.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

PreparedBackendManifest
BuildVulkanBackendManifest(const KernelExecutionStep &step,
                           const rund::kernel::ComputePlan &plan,
                           const BoundStep *const bound,
                           const std::uint64_t max_dispatch_groups) noexcept {
  PreparedBackendManifest manifest{};
  const bool has_checks = !step.artifact.metadata.read_routes.empty();
  const bool controlled = (bound != nullptr ? bound->control.active()
                                            : (step.control.has_count() ||
                                               step.control.has_predicate())) ||
                          has_checks;
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map:
    if (!BuildVulkanMapManifest(step, plan, bound, has_checks, controlled,
                                manifest)) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::Scan:
  case rund::kernel::NodeKind::SegmentedScan:
  case rund::kernel::NodeKind::Partition:
    if (!BuildVulkanScanManifest(step, plan, manifest)) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::SegmentedReduce:
  case rund::kernel::NodeKind::Reduce:
  case rund::kernel::NodeKind::ScatterReduce:
    if (!BuildVulkanReductionManifest(step, plan, manifest)) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::Sort:
    if (!BuildVulkanSortManifest(step, plan, manifest)) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::Compact:
  case rund::kernel::NodeKind::Gather:
  case rund::kernel::NodeKind::Histogram:
    if (!BuildVulkanCollectiveManifest(step, manifest)) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::Scatter:
    if (!BuildVulkanScatterManifest(step, manifest)) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::Stencil:
  case rund::kernel::NodeKind::Window:
    if (!BuildVulkanRangeManifest(step, manifest)) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
  case rund::kernel::NodeKind::Factor:
  case rund::kernel::NodeKind::Solve:
  case rund::kernel::NodeKind::Spectrum:
    if (!BuildVulkanNumericManifest(step, manifest)) {
      return manifest;
    }
    break;
  }
  const bool complete = PlanVulkanCaptureManifest(
                            step, plan, bound, max_dispatch_groups, manifest) &&
                        CompleteVulkanBackendManifest(manifest);
  (void)complete;
  return manifest;
}

#else

PreparedBackendManifest
BuildVulkanBackendManifest(const KernelExecutionStep &,
                           const rund::kernel::ComputePlan &, const BoundStep *,
                           std::uint64_t) noexcept {
  return {};
}

#endif

} // namespace rund::node::accel::detail
