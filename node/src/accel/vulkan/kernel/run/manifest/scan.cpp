#include "internal.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../../../kernel.hpp"
#include "../../../partition/local.hpp"
#include "../../../scan/local.hpp"
#include "../../../scan/source.hpp"
#include "../../../segmented/local.hpp"
#include "../../pipeline/source.hpp"

#include <kernel/program/compute/scan/plan.hpp>

#include <cstdint>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] std::uint64_t ScanStages(const std::uint64_t passes) noexcept {
  return passes == 1u ? std::uint64_t{1u} : std::uint64_t{3u};
}

} // namespace

bool BuildVulkanScanManifest(const KernelExecutionStep &step,
                             const rund::kernel::ComputePlan &plan,
                             PreparedBackendManifest &manifest) noexcept {
  switch (step.kind()) {
  case rund::kernel::NodeKind::Scan: {
    const auto &active = step.operation.get<operation::Scan>();
    const RangePrefixExec prefix = PlanScanPrefixExecution(active.plan);
    if (!prefix.ok()) {
      return false;
    }
    const std::uint64_t stages = prefix.stage_count();
    manifest = PreparedBackendManifest{
        .source_build_count = stages,
        .source_library_dependency_count = stages,
        .pipeline_stage_count = stages,
        .descriptor_set_count = stages,
        .descriptor_binding_count = 6u * stages,
        .descriptor_lease_count = stages,
        .descriptor_dependency_count = stages,
    };
    const bool inclusive = active.desc.op == rund::kernel::ScanOp::InclusiveSum;
    const auto add_stage = [&](const VulkanScanStage stage) noexcept {
      std::uint64_t source_bytes = 0u;
      return VulkanScanSourceBytes(active.desc.element, plan.domain, stage,
                                   inclusive, source_bytes) &&
             AddPreparedBackendCacheDependency(
                 manifest,
                 PreparedBackendCacheDependency{
                     .source_recipe = 0x76756c6b2e736e00ull +
                                      static_cast<std::uint64_t>(stage) + 1u,
                     .source_upper_bytes = source_bytes,
                     .pipeline_stage_count = 1u,
                 });
    };
    return add_stage(VulkanScanStage::Block) &&
           (stages == 1u || (add_stage(VulkanScanStage::Prefix) &&
                             add_stage(VulkanScanStage::Offset)));
  }
  case rund::kernel::NodeKind::SegmentedScan: {
    const auto &active = step.operation.get<operation::SegmentedScan>();
    const std::uint64_t stages = ScanStages(active.plan.pass_count);
    manifest = PreparedBackendManifest{
        .source_build_count = stages,
        .source_library_dependency_count = stages,
        .pipeline_stage_count = stages,
        .descriptor_set_count = stages,
        .descriptor_binding_count = 7u * stages,
        .descriptor_lease_count = stages,
        .descriptor_dependency_count = stages,
    };
    const auto add_stage = [&](const VulkanSegmentedScanStage stage) noexcept {
      std::uint64_t source_bytes = 0u;
      return VulkanSegmentedScanSourceBytes(active.desc.element, plan.domain,
                                            stage, source_bytes) &&
             AddPreparedBackendCacheDependency(
                 manifest,
                 PreparedBackendCacheDependency{
                     .source_recipe = 0x76756c6b2e736700ull +
                                      static_cast<std::uint64_t>(stage) + 1u,
                     .source_upper_bytes = source_bytes,
                     .pipeline_stage_count = 1u,
                 });
    };
    return add_stage(VulkanSegmentedScanStage::Block) &&
           (stages == 1u || (add_stage(VulkanSegmentedScanStage::Prefix) &&
                             add_stage(VulkanSegmentedScanStage::Offset)));
  }
  case rund::kernel::NodeKind::Partition: {
    const auto &active = step.operation.get<operation::Partition>();
    const rund::kernel::ScanPlan scan_plan =
        rund::kernel::PlanScan(rund::kernel::ScanDesc{
            .op = rund::kernel::ScanOp::ExclusiveSum,
            .element = rund::kernel::ScanElement::U32,
            .element_count = active.plan.element_count,
            .block_size = block::VulkanPartition,
        });
    const RangePrefixExec prefix = PlanScanPrefixExecution(scan_plan);
    if (!prefix.ok()) {
      return false;
    }
    const std::uint64_t scan = prefix.stage_count();
    manifest.source_build_count = 2u + scan;
    manifest.source_library_dependency_count = 2u + scan;
    manifest.pipeline_stage_count = 2u + scan;
    manifest.descriptor_set_count = 2u + scan;
    manifest.descriptor_binding_count = 8u + 6u * scan;
    manifest.descriptor_lease_count = manifest.descriptor_set_count;
    manifest.descriptor_dependency_count = manifest.pipeline_stage_count;
    for (const PartitionStage stage :
         {PartitionStage::Classify, PartitionStage::Scatter}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanPartitionSourceBytes(stage, active.desc.flag_bytes,
                                      active.desc.value_bytes, source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest,
              PreparedBackendCacheDependency{
                  .source_recipe = 0x76756c6b2e707400ull +
                                   static_cast<std::uint64_t>(stage) + 1u,
                  .source_upper_bytes = source_bytes,
                  .pipeline_stage_count = 1u,
              })) {
        return false;
      }
    }
    const auto add_stage = [&](const VulkanScanStage stage) noexcept {
      std::uint64_t source_bytes = 0u;
      return VulkanScanSourceBytes(rund::kernel::ScanElement::U32,
                                   rund::kernel::ComputeDomain::U32, stage,
                                   false, source_bytes) &&
             AddPreparedBackendCacheDependency(
                 manifest,
                 PreparedBackendCacheDependency{
                     .source_recipe = 0x76756c6b2e707300ull +
                                      static_cast<std::uint64_t>(stage) + 1u,
                     .source_upper_bytes = source_bytes,
                     .pipeline_stage_count = 1u,
                 });
    };
    return add_stage(VulkanScanStage::Block) &&
           (scan == 1u || (add_stage(VulkanScanStage::Prefix) &&
                           add_stage(VulkanScanStage::Offset)));
  }
  default:
    return false;
  }
}

} // namespace rund::node::accel::detail

#endif
