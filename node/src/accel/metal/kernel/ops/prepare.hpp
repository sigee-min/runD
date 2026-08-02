#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../../compact.hpp"
#include "../../../gather.hpp"
#include "../../../histogram.hpp"
#include "../../../partition.hpp"
#include "../../../reduce/metal.hpp"
#include "../../../scan/metal.hpp"
#include "../../../scatter.hpp"
#include "../../../segmented/metal.hpp"
#include "../../../segmented/reduce/metal.hpp"
#include "../../../sort.hpp"
#include "../../../stencil.hpp"
#include "../../numeric.hpp"
#include "../../runtime/map/api.hpp"
#include "../../scan/kernel/local.hpp"
#include "../../sort/local.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] inline rund::AccelCheck
PrepareMetalMapStep(const rund::AccelDevice &pick, const BoundStep &step,
                    const MetalKernelImmutablePipelines *,
                    std::shared_ptr<void> &resources) {
  const StepBinds *const bindings =
      BindingsFor<StepBinds>(step, rund::kernel::NodeKind::Map);
  if (bindings == nullptr || step.planned->artifact == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  return PrepareMetalMap(pick, step.planned->plan, *step.planned->artifact,
                         step.map_windows.data(), step.map_windows.size(),
                         MapBindingFor(step), step.control, resources);
}

[[nodiscard]] inline rund::AccelCheck
PrepareMetalScanStep(const rund::AccelDevice &pick, const BoundStep &step,
                     const MetalKernelImmutablePipelines *pipelines,
                     std::shared_ptr<void> &resources) {
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
      PrepareMetalScan(pick, active->desc, active->plan, step.planned->domain,
                       *bindings, resources, pipelines);
  if (check.ok) {
    static_cast<MetalScanEncodeResources *>(resources.get())->control =
        step.control.control;
  }
  return check;
}

[[nodiscard]] inline rund::AccelCheck
PrepareMetalSegmentedStep(const rund::AccelDevice &pick, const BoundStep &step,
                          const MetalKernelImmutablePipelines *pipelines,
                          std::shared_ptr<void> &resources) {
  const SegmentedScanBinds *const bindings = BindingsFor<SegmentedScanBinds>(
      step, rund::kernel::NodeKind::SegmentedScan);
  const auto *active = OperationFor<operation::SegmentedScan>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareMetalSegmentedScan(pick, active->desc, active->plan,
                                         step.planned->domain, *bindings,
                                         resources, pipelines);
}

[[nodiscard]] inline rund::AccelCheck
PrepareMetalSegmentedReduceStep(const rund::AccelDevice &pick,
                                const BoundStep &step,
                                const MetalKernelImmutablePipelines *pipelines,
                                std::shared_ptr<void> &resources) {
  const SegmentedReduceBinds *const bindings =
      BindingsFor<SegmentedReduceBinds>(
          step, rund::kernel::NodeKind::SegmentedReduce);
  const auto *active = OperationFor<operation::SegmentedReduce>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareMetalSegmentedReduce(pick, active->desc, active->plan,
                                           step.planned->domain, *bindings,
                                           resources, pipelines);
}

[[nodiscard]] inline rund::AccelCheck
PrepareMetalSortStep(const rund::AccelDevice &pick, const BoundStep &step,
                     const MetalKernelImmutablePipelines *pipelines,
                     std::shared_ptr<void> &resources) {
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
      PrepareMetalSort(pick, active->desc, active->plan, step.planned->domain,
                       *bindings, resources, pipelines);
  if (check.ok) {
    static_cast<MetalSortEncodeResources *>(resources.get())->control =
        step.control.control;
  }
  return check;
}

[[nodiscard]] inline rund::AccelCheck
PrepareMetalCompactStep(const rund::AccelDevice &pick, const BoundStep &step,
                        const MetalKernelImmutablePipelines *pipelines,
                        std::shared_ptr<void> &resources) {
  const CompactBinds *const bindings =
      BindingsFor<CompactBinds>(step, rund::kernel::NodeKind::Compact);
  const auto *active = OperationFor<operation::Compact>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareMetalCompact(pick, active->desc, active->plan, *bindings,
                                   resources, pipelines);
}

[[nodiscard]] inline rund::AccelCheck
PrepareMetalGatherStep(const rund::AccelDevice &pick, const BoundStep &step,
                       const MetalKernelImmutablePipelines *pipelines,
                       std::shared_ptr<void> &resources) {
  const GatherBinds *const bindings =
      BindingsFor<GatherBinds>(step, rund::kernel::NodeKind::Gather);
  const auto *active = OperationFor<operation::Gather>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareMetalGather(pick, active->desc, active->plan, *bindings,
                                  resources, pipelines);
}

[[nodiscard]] inline rund::AccelCheck
PrepareMetalHistogramStep(const rund::AccelDevice &pick, const BoundStep &step,
                          const MetalKernelImmutablePipelines *pipelines,
                          std::shared_ptr<void> &resources) {
  const HistogramBinds *const bindings =
      BindingsFor<HistogramBinds>(step, rund::kernel::NodeKind::Histogram);
  const auto *active = OperationFor<operation::Histogram>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareMetalHistogram(pick, active->desc, active->plan,
                                     *bindings, resources, pipelines);
}

[[nodiscard]] inline rund::AccelCheck
PrepareMetalScatterReduceStep(const rund::AccelDevice &pick,
                              const BoundStep &step,
                              const MetalKernelImmutablePipelines *pipelines,
                              std::shared_ptr<void> &resources) {
  const ScatterReduceBinds *const bindings = BindingsFor<ScatterReduceBinds>(
      step, rund::kernel::NodeKind::ScatterReduce);
  const auto *active = OperationFor<operation::ScatterReduce>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareMetalScatterReduce(pick, active->plan, *bindings,
                                         resources, pipelines);
}

[[nodiscard]] inline rund::AccelCheck
PrepareMetalPartitionStep(const rund::AccelDevice &pick, const BoundStep &step,
                          const MetalKernelImmutablePipelines *pipelines,
                          std::shared_ptr<void> &resources) {
  const PartitionBinds *const bindings =
      BindingsFor<PartitionBinds>(step, rund::kernel::NodeKind::Partition);
  const auto *active = OperationFor<operation::Partition>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareMetalPartition(pick, active->desc, active->plan,
                                     *bindings, resources, pipelines);
}

[[nodiscard]] inline rund::AccelCheck
PrepareMetalReduceStep(const rund::AccelDevice &pick, const BoundStep &step,
                       const MetalKernelImmutablePipelines *pipelines,
                       std::shared_ptr<void> &resources) {
  const ReduceBinds *const bindings =
      BindingsFor<ReduceBinds>(step, rund::kernel::NodeKind::Reduce);
  const auto *active = OperationFor<operation::Reduce>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareMetalReduce(pick, active->desc, active->plan,
                                  step.planned->domain, *bindings, resources,
                                  pipelines);
}

[[nodiscard]] inline rund::AccelCheck
PrepareMetalScatterStep(const rund::AccelDevice &pick, const BoundStep &step,
                        const MetalKernelImmutablePipelines *pipelines,
                        std::shared_ptr<void> &resources) {
  const ScatterBinds *const bindings =
      BindingsFor<ScatterBinds>(step, rund::kernel::NodeKind::Scatter);
  const auto *active = OperationFor<operation::Scatter>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareMetalScatter(pick, active->desc, active->plan, *bindings,
                                   resources, pipelines);
}

[[nodiscard]] inline rund::AccelCheck
PrepareMetalStencilStep(const rund::AccelDevice &pick, const BoundStep &step,
                        const MetalKernelImmutablePipelines *pipelines,
                        std::shared_ptr<void> &resources) {
  const StencilBinds *const bindings =
      BindingsFor<StencilBinds>(step, rund::kernel::NodeKind::Stencil);
  const auto *active = OperationFor<operation::Stencil>(step);
  return bindings == nullptr || active == nullptr
             ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
             : PrepareMetalStencil(pick, active->desc, active->plan,
                                   step.planned->domain, *bindings, resources,
                                   pipelines);
}

#include "prepare/numeric.hpp"

#endif

} // namespace rund::node::accel::detail
