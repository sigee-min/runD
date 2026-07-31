#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../../compact.hpp"
#include "../../../gather.hpp"
#include "../../../histogram.hpp"
#include "../../../partition.hpp"
#include "../../../reduce/vulkan.hpp"
#include "../../../scan/vulkan.hpp"
#include "../../../scatter.hpp"
#include "../../../segmented/reduce/vulkan.hpp"
#include "../../../segmented/vulkan.hpp"
#include "../../../sort.hpp"
#include "../../../stencil.hpp"
#include "../../map/api.hpp"
#include "../../numeric.hpp"
#include "../../scan/kernel/local.hpp"
#include "../../sort/local/api.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] inline rund::AccelCheck
PrepareVulkanMapStep(const rund::AccelDevice &pick, const BoundStep &step,
                     const KernelPreparationMode mode,
                     std::shared_ptr<void> &resources) {
  (void)mode;
  const StepBinds *const bindings =
      BindingsFor<StepBinds>(step, rund::kernel::NodeKind::Map);
  if (bindings == nullptr || step.planned->artifact == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  return PrepareVulkanMap(pick, step.planned->plan, *step.planned->artifact,
                          step.map_windows.data(), step.map_windows.size(),
                          MapBindingFor(step), step.control, resources);
}

[[nodiscard]] inline rund::AccelCheck
PrepareVulkanScanStep(const rund::AccelDevice &pick, const BoundStep &step,
                      const KernelPreparationMode mode,
                      std::shared_ptr<void> &resources) {
  (void)mode;
  const ScanBinds *const bindings =
      BindingsFor<ScanBinds>(step, rund::kernel::NodeKind::Scan);
  if (bindings == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const auto *active = OperationFor<operation::Scan>(step);
  if (active == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const rund::AccelCheck check =
      PrepareVulkanScan(pick, active->desc, active->plan, step.planned->domain,
                        *bindings, resources);
  if (check.ok) {
    static_cast<VulkanKernelScanResources *>(resources.get())->control =
        step.control.control;
  }
  return check;
}

[[nodiscard]] inline rund::AccelCheck
PrepareVulkanSegmentedStep(const rund::AccelDevice &pick, const BoundStep &step,
                           const KernelPreparationMode mode,
                           std::shared_ptr<void> &resources) {
  (void)mode;
  const SegmentedScanBinds *const bindings = BindingsFor<SegmentedScanBinds>(
      step, rund::kernel::NodeKind::SegmentedScan);
  const auto *active = OperationFor<operation::SegmentedScan>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareVulkanSegmentedScan(pick, active->desc, active->plan,
                                          step.planned->domain, *bindings,
                                          resources);
}

[[nodiscard]] inline rund::AccelCheck PrepareVulkanSegmentedReduceStep(
    const rund::AccelDevice &pick, const BoundStep &step,
    const KernelPreparationMode mode, std::shared_ptr<void> &resources) {
  (void)mode;
  const SegmentedReduceBinds *const bindings =
      BindingsFor<SegmentedReduceBinds>(
          step, rund::kernel::NodeKind::SegmentedReduce);
  const auto *active = OperationFor<operation::SegmentedReduce>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareVulkanSegmentedReduce(pick, active->desc, active->plan,
                                            step.planned->domain, *bindings,
                                            resources);
}

[[nodiscard]] inline rund::AccelCheck
PrepareVulkanSortStep(const rund::AccelDevice &pick, const BoundStep &step,
                      const KernelPreparationMode mode,
                      std::shared_ptr<void> &resources) {
  (void)mode;
  const SortBinds *const bindings =
      BindingsFor<SortBinds>(step, rund::kernel::NodeKind::Sort);
  if (bindings == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const auto *active = OperationFor<operation::Sort>(step);
  if (active == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const rund::AccelCheck check =
      PrepareVulkanSort(pick, active->desc, active->plan, step.planned->domain,
                        *bindings, resources);
  if (check.ok) {
    static_cast<VulkanSortEncodeResources *>(resources.get())->control =
        step.control.control;
  }
  return check;
}

[[nodiscard]] inline rund::AccelCheck
PrepareVulkanCompactStep(const rund::AccelDevice &pick, const BoundStep &step,
                         const KernelPreparationMode mode,
                         std::shared_ptr<void> &resources) {
  (void)mode;
  const CompactBinds *const bindings =
      BindingsFor<CompactBinds>(step, rund::kernel::NodeKind::Compact);
  const auto *active = OperationFor<operation::Compact>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareVulkanCompact(pick, active->desc, active->plan, *bindings,
                                    resources);
}

[[nodiscard]] inline rund::AccelCheck
PrepareVulkanGatherStep(const rund::AccelDevice &pick, const BoundStep &step,
                        const KernelPreparationMode mode,
                        std::shared_ptr<void> &resources) {
  (void)mode;
  const GatherBinds *const bindings =
      BindingsFor<GatherBinds>(step, rund::kernel::NodeKind::Gather);
  const auto *active = OperationFor<operation::Gather>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareVulkanGather(pick, active->desc, active->plan, *bindings,
                                   resources);
}

[[nodiscard]] inline rund::AccelCheck
PrepareVulkanHistogramStep(const rund::AccelDevice &pick, const BoundStep &step,
                           const KernelPreparationMode mode,
                           std::shared_ptr<void> &resources) {
  (void)mode;
  const HistogramBinds *const bindings =
      BindingsFor<HistogramBinds>(step, rund::kernel::NodeKind::Histogram);
  const auto *active = OperationFor<operation::Histogram>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareVulkanHistogram(pick, active->desc, active->plan,
                                      *bindings, resources);
}

[[nodiscard]] inline rund::AccelCheck PrepareVulkanScatterReduceStep(
    const rund::AccelDevice &pick, const BoundStep &step,
    const KernelPreparationMode mode, std::shared_ptr<void> &resources) {
  const ScatterReduceBinds *const bindings = BindingsFor<ScatterReduceBinds>(
      step, rund::kernel::NodeKind::ScatterReduce);
  const auto *active = OperationFor<operation::ScatterReduce>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareVulkanScatterReduce(pick, active->plan, *bindings, mode,
                                          resources);
}

[[nodiscard]] inline rund::AccelCheck
PrepareVulkanPartitionStep(const rund::AccelDevice &pick, const BoundStep &step,
                           const KernelPreparationMode mode,
                           std::shared_ptr<void> &resources) {
  (void)mode;
  const PartitionBinds *const bindings =
      BindingsFor<PartitionBinds>(step, rund::kernel::NodeKind::Partition);
  const auto *active = OperationFor<operation::Partition>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareVulkanPartition(pick, active->desc, active->plan,
                                      *bindings, resources);
}

#include "prepare/collective.hpp"

#endif

} // namespace rund::node::accel::detail
