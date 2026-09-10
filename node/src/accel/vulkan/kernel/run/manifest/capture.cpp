#include "../../../../kernel/backend/execute.hpp"
#include "../../../../kernel/recurrence/plan.hpp"
#include "../../../../kernel/status.hpp"
#include "../../../../resident/window/admission/runtime/windows.hpp"

#include "../../../collective/chunk.hpp"
#include "../../../compact/local.hpp"
#include "../../../descriptor.hpp"
#include "../../../gather/local.hpp"
#include "../../../histogram/local.hpp"
#include "../../../kernel.hpp"
#include "../../../map/api.hpp"
#include "../../../map/local.hpp"
#include "../../../map/source/upper.hpp"
#include "../../../numeric/source.hpp"
#include "../../../numeric/state.hpp"
#include "../../../partition/local.hpp"
#include "../../../range/local.hpp"
#include "../../../reduce/local.hpp"
#include "../../../scan/local.hpp"
#include "../../../scan/source.hpp"
#include "../../../scatter/local.hpp"
#include "../../../scatter/reduce/model.hpp"
#include "../../../segmented/local.hpp"
#include "../../../segmented/reduce/model.hpp"
#include "../../../sort/local/state.hpp"
#include "../../manifest.hpp"
#include "../../ops/prepare.hpp"
#include "../../pipeline/capacity.hpp"
#include "../../pipeline/evidence.hpp"
#include "../../pipeline/recurrence.hpp"
#include "../../pipeline/source.hpp"
#include "../../pipeline/state.hpp"
#include "../../reset/source.hpp"

#include "../../../../primitive/block.hpp"
#include "../../../../sort/block/vulkan.hpp"

#include <kernel/program/compute/scan/plan.hpp>

#include <limits>

#include "../route.hpp"
#include "capture.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool PlanVulkanCaptureManifest(
    const KernelExecutionStep &step, const rund::kernel::ComputePlan &plan,
    const BoundStep *const bound, const std::uint64_t max_dispatch_groups,
    PreparedBackendManifest &manifest) noexcept {
  using rund::kernel::checked::add;
  using rund::kernel::checked::mul;
  if (max_dispatch_groups == 0u || plan.dispatch_count == 0u) {
    return false;
  }
  const bool has_checks = !step.artifact.metadata.read_routes.empty();
  const bool controlled = (bound != nullptr ? bound->control.active()
                                            : (step.control.has_count() ||
                                               step.control.has_predicate())) ||
                          has_checks;
  const std::uint32_t control_iteration =
      bound == nullptr ? step.control.iteration
                       : bound->control.control.iteration;
  std::uint64_t direct = 0u;
  std::uint64_t indirect = 0u;
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map:
    direct = VulkanMapRouteDispatches(plan, bound);
    if (controlled && has_checks && !add(direct, 1u, direct)) {
      return false;
    }
    indirect = controlled ? VulkanMapRouteDispatches(plan, bound) : 0u;
    break;
  case rund::kernel::NodeKind::Scan: {
    const auto &active = step.operation.get<operation::Scan>().plan;
    direct = ScanPrefixDispatches(PlanScanPrefixExecution(active),
                                  max_dispatch_groups);
    break;
  }
  case rund::kernel::NodeKind::SegmentedScan: {
    const auto &active = step.operation.get<operation::SegmentedScan>().plan;
    direct = ScanDispatches(active.pass_count, active.block_count,
                            max_dispatch_groups);
    break;
  }
  case rund::kernel::NodeKind::SegmentedReduce:
    direct = 3u;
    indirect = 1u;
    break;
  case rund::kernel::NodeKind::Sort: {
    const auto &active = step.operation.get<operation::Sort>().plan;
    const std::uint64_t blocks =
        CeilGroups(active.element_count, kVulkanSortBlockSize);
    const std::uint64_t chunks = CeilGroups(blocks, max_dispatch_groups);
    const std::uint64_t fixed = blocks == 1u ? 1u : 2u;
    std::uint64_t pass_direct = 0u;
    std::uint64_t pass_indirect = 0u;
    if (blocks == 0u || chunks == 0u || active.radix_pass_count == 0u ||
        !mul(active.radix_pass_count, fixed, pass_direct) ||
        !add(pass_direct, 1u, direct) ||
        !mul(active.radix_pass_count, chunks, pass_indirect) ||
        !mul(pass_indirect, 2u, indirect)) {
      return false;
    }
    break;
  }
  case rund::kernel::NodeKind::Gather:
    direct = 1u;
    indirect = 1u;
    break;
  case rund::kernel::NodeKind::Partition: {
    const auto &active = step.operation.get<operation::Partition>().plan;
    const rund::kernel::ScanPlan scan_plan =
        rund::kernel::PlanScan(rund::kernel::ScanDesc{
            .op = rund::kernel::ScanOp::ExclusiveSum,
            .element = rund::kernel::ScanElement::U32,
            .element_count = active.element_count,
            .block_size = block::VulkanPartition,
        });
    const std::uint64_t scan = ScanPrefixDispatches(
        PlanScanPrefixExecution(scan_plan), max_dispatch_groups);
    if (scan == 0u || !add(scan, 2u, direct)) {
      return false;
    }
    break;
  }
  case rund::kernel::NodeKind::ScatterReduce:
    direct = 1u;
    indirect = 2u;
    break;
  case rund::kernel::NodeKind::Window: {
    const RangePlan *const range = RangePlanFor(step.operation);
    if (range == nullptr || !range->ok()) {
      return false;
    }
    if (range->shape().resident_counted()) {
      direct = 1u;
      indirect = range->stage_count();
    } else {
      direct = plan.dispatch_count;
    }
    break;
  }
  default:
    direct = plan.dispatch_count;
    break;
  }
  if (direct == 0u) {
    return false;
  }
  manifest.capture_direct_dispatch_count = direct;
  manifest.capture_indirect_dispatch_count = indirect;
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map:
    manifest.status_source_count = controlled ? 1u : 0u;
    manifest.telemetry_source_count = controlled ? 1u : 0u;
    break;
  case rund::kernel::NodeKind::Stencil:
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
    break;
  case rund::kernel::NodeKind::Window: {
    const RangePlan *const range = RangePlanFor(step.operation);
    if (range != nullptr && range->ok() && range->shape().resident_counted()) {
      manifest.status_source_count = 1u;
      manifest.telemetry_source_count = 1u;
    }
    break;
  }
  default:
    manifest.status_source_count = 1u;
    break;
  }
  if (step.kind() == rund::kernel::NodeKind::Gather ||
      step.kind() == rund::kernel::NodeKind::ScatterReduce ||
      ((step.kind() == rund::kernel::NodeKind::Scan ||
        step.kind() == rund::kernel::NodeKind::Sort) &&
       control_iteration != 0u)) {
    manifest.telemetry_source_count = 1u;
  }
  switch (step.kind()) {
  case rund::kernel::NodeKind::Factor:
    manifest.status_entry_count =
        step.operation.get<operation::Factor>().plan.status_count;
    break;
  case rund::kernel::NodeKind::Solve:
    manifest.status_entry_count =
        step.operation.get<operation::Solve>().plan.status_count;
    break;
  case rund::kernel::NodeKind::Spectrum:
    manifest.status_entry_count =
        step.operation.get<operation::Spectrum>().plan.status_count;
    break;
  default:
    manifest.status_entry_count = manifest.status_source_count;
    break;
  }
  if ((manifest.status_source_count == 0u) !=
      (manifest.status_entry_count == 0u)) {
    return false;
  }
  if (!mul(manifest.status_source_count, 2u, manifest.status_command_count) ||
      !mul(manifest.status_source_count,
           VulkanPipelineStatusSourceParameterBytes,
           manifest.status_parameter_bytes)) {
    return false;
  }
  return true;
}

#endif

} // namespace rund::node::accel::detail
