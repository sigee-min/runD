#include "../../../../kernel/backend/execute.hpp"
#include "../../../../kernel/backend/template_plan.hpp"
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
#include "../../../map/source_upper.hpp"
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
#include "../../reset_source.hpp"

#include "../../../../primitive/block.hpp"
#include "../../../../sort/block/vulkan.hpp"

#include <kernel/program/compute/scan/plan.hpp>

#include <limits>

#include "../route.hpp"
#include "capture.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool VulkanNumericSourceBytes(const rund::kernel::NodeKind kind,
                                            const bool wide,
                                            std::uint64_t &bytes) noexcept {
  switch (kind) {
  case rund::kernel::NodeKind::Transform:
    return wide ? TransformSource64Bytes(bytes) : TransformSourceBytes(bytes);
  case rund::kernel::NodeKind::Matrix:
    return wide ? MatrixSource64Bytes(bytes) : MatrixSourceBytes(bytes);
  case rund::kernel::NodeKind::Factor:
    return wide ? FactorSource64Bytes(bytes) : FactorSourceBytes(bytes);
  case rund::kernel::NodeKind::Solve:
    return wide ? SolveSource64Bytes(bytes) : SolveSourceBytes(bytes);
  case rund::kernel::NodeKind::Spectrum:
    return wide ? SpectrumSource64Bytes(bytes) : SpectrumSourceBytes(bytes);
  default:
    return false;
  }
}

} // namespace

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
  const auto scan_stages = [](const std::uint64_t passes) {
    return passes == 1u ? std::uint64_t{1u} : std::uint64_t{3u};
  };
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map: {
    const std::uint64_t route_dispatches =
        VulkanMapRouteDispatches(plan, bound);
    const std::uint64_t checks = UniqueVulkanMapCheckCount(step.artifact);
    const std::uint64_t check_stage = has_checks ? 1u : 0u;
    const std::uint64_t control_stage = controlled ? 1u : 0u;
    manifest.source_build_count = 1u + control_stage + check_stage;
    manifest.source_library_dependency_count = manifest.source_build_count;
    manifest.pipeline_stage_count = manifest.source_build_count;
    manifest.descriptor_dependency_count = manifest.pipeline_stage_count;
    manifest.descriptor_lease_count = control_stage + check_stage;
    std::uint64_t bindings_per_window = 0u;
    std::uint64_t window_bindings = 0u;
    std::uint64_t check_bindings = 0u;
    if (!rund::kernel::checked::add(plan.input_buffer_count,
                                    plan.output_buffer_count,
                                    bindings_per_window) ||
        !rund::kernel::checked::add(bindings_per_window, 1u + control_stage,
                                    bindings_per_window) ||
        !rund::kernel::checked::mul(route_dispatches, bindings_per_window,
                                    window_bindings) ||
        !rund::kernel::checked::add(checks, 3u, check_bindings) ||
        !rund::kernel::checked::mul(check_bindings, check_stage,
                                    check_bindings) ||
        !rund::kernel::checked::add(route_dispatches, control_stage,
                                    manifest.descriptor_set_count) ||
        !rund::kernel::checked::add(manifest.descriptor_set_count, check_stage,
                                    manifest.descriptor_set_count) ||
        !rund::kernel::checked::add(window_bindings, 4u * control_stage,
                                    manifest.descriptor_binding_count) ||
        !rund::kernel::checked::add(manifest.descriptor_binding_count,
                                    check_bindings,
                                    manifest.descriptor_binding_count)) {
      return manifest;
    }
    std::uint64_t main_source = 0u;
    std::uint64_t final_main_source = 0u;
    std::uint64_t raw_source_transient = 0u;
    if (!backend_template_plan::map_source_upper(step, plan, main_source,
                                                 raw_source_transient) ||
        (controlled && !VulkanControlledMapSourceUpperBytes(
                           plan, main_source, final_main_source)) ||
        !AddPreparedBackendCacheDependency(
            manifest, PreparedBackendCacheDependency{
                          .source_recipe = 0x76756c6b2e6d6170ull,
                          .source_upper_bytes =
                              controlled ? final_main_source : main_source,
                          .pipeline_stage_count = 1u,
                      })) {
      return manifest;
    }
    if (controlled &&
        !backend_source_recipe::string_external_storage_upper_bytes(
            main_source, raw_source_transient)) {
      return manifest;
    }
    manifest.cold_source_transient_bytes = raw_source_transient;
    if (controlled) {
      std::uint64_t control_source = 0u;
      if (!VulkanMapControlSourceBytes(control_source) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = 0x76756c6b2e637472ull,
                            .source_upper_bytes = control_source,
                            .pipeline_stage_count = 1u,
                        })) {
        return manifest;
      }
    }
    if (has_checks) {
      std::uint64_t check_source = 0u;
      if (!VulkanMapCheckSourceUpperBytes(step.artifact, check_source) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = 0x76756c6b2e63686bull,
                            .source_upper_bytes = check_source,
                            .pipeline_stage_count = 1u,
                        })) {
        return manifest;
      }
    }
    // The main Map artifact is moved into the cached pipeline. Control and
    // bounds-check collective pipelines copy their source, so their caller
    // storage is the only additional full-source allocation live during cache
    // publication. Dependency zero is always the moved main artifact.
    for (std::size_t index = 1u; index < manifest.cache_dependency_entry_count;
         ++index) {
      manifest.cold_source_transient_bytes = std::max(
          manifest.cold_source_transient_bytes,
          manifest.source_dependencies[index].source_storage_upper_bytes);
    }
    break;
  }
  case rund::kernel::NodeKind::Scan: {
    const auto &active = step.operation.get<operation::Scan>();
    const RangePrefixExec prefix = PlanScanPrefixExecution(active.plan);
    if (!prefix.ok()) {
      return manifest;
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
    const auto add_scan_stage = [&](const VulkanScanStage stage) noexcept {
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
    if (!add_scan_stage(VulkanScanStage::Block) ||
        (stages != 1u && (!add_scan_stage(VulkanScanStage::Prefix) ||
                          !add_scan_stage(VulkanScanStage::Offset)))) {
      return manifest;
    }
    break;
  }
  case rund::kernel::NodeKind::SegmentedScan: {
    const auto &active = step.operation.get<operation::SegmentedScan>();
    const std::uint64_t stages = scan_stages(active.plan.pass_count);
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
    if (!add_stage(VulkanSegmentedScanStage::Block) ||
        (stages != 1u && (!add_stage(VulkanSegmentedScanStage::Prefix) ||
                          !add_stage(VulkanSegmentedScanStage::Offset)))) {
      return manifest;
    }
    break;
  }
  case rund::kernel::NodeKind::SegmentedReduce: {
    manifest = PreparedBackendManifest{.source_build_count = 4u,
                                       .source_library_dependency_count = 4u,
                                       .pipeline_stage_count = 4u,
                                       .descriptor_set_count = 4u,
                                       .descriptor_binding_count = 24u,
                                       .descriptor_lease_count = 4u,
                                       .descriptor_dependency_count = 4u};
    const auto &active = step.operation.get<operation::SegmentedReduce>();
    for (const VulkanSegmentedReduceStage stage :
         {VulkanSegmentedReduceStage::Classify,
          VulkanSegmentedReduceStage::Prefix,
          VulkanSegmentedReduceStage::Scatter,
          VulkanSegmentedReduceStage::Reduce}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanSegmentedReduceSourceBytes(active.plan, plan.domain, stage,
                                            source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = 0x76756c6b2e727300ull +
                                             static_cast<std::uint64_t>(stage),
                            .source_upper_bytes = source_bytes,
                            .pipeline_stage_count = 1u,
                        })) {
        return manifest;
      }
    }
    break;
  }
  case rund::kernel::NodeKind::Sort: {
    const auto &active = step.operation.get<operation::Sort>();
    const std::uint64_t passes = active.plan.radix_pass_count;
    if (!rund::kernel::checked::mul(4u, passes,
                                    manifest.descriptor_set_count) ||
        !rund::kernel::checked::add(manifest.descriptor_set_count, 1u,
                                    manifest.descriptor_set_count) ||
        !rund::kernel::checked::mul(9u, manifest.descriptor_set_count,
                                    manifest.descriptor_binding_count)) {
      return manifest;
    }
    manifest.source_build_count = 5u;
    manifest.source_library_dependency_count = 5u;
    manifest.pipeline_stage_count = 5u;
    manifest.descriptor_lease_count = manifest.descriptor_set_count;
    manifest.descriptor_dependency_count = 5u;
    for (const SortStage stage :
         {SortStage::Dispatch, SortStage::Classify, SortStage::Prefix,
          SortStage::Base, SortStage::Scatter}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanSortSourceBytes(active.desc.key, stage, source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest,
              PreparedBackendCacheDependency{
                  .source_recipe = 0x76756c6b2e736f00ull +
                                   static_cast<std::uint64_t>(stage) + 1u,
                  .source_upper_bytes = source_bytes,
                  .pipeline_stage_count = 1u,
              })) {
        return manifest;
      }
    }
    break;
  }
  case rund::kernel::NodeKind::Compact: {
    manifest = PreparedBackendManifest{.source_build_count = 3u,
                                       .source_library_dependency_count = 3u,
                                       .pipeline_stage_count = 3u,
                                       .descriptor_set_count = 3u,
                                       .descriptor_binding_count = 18u,
                                       .descriptor_lease_count = 3u,
                                       .descriptor_dependency_count = 3u};
    for (const CompactStage stage :
         {CompactStage::Classify, CompactStage::Prefix,
          CompactStage::Scatter}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanCompactSourceBytes(stage, source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest,
              PreparedBackendCacheDependency{
                  .source_recipe = 0x76756c6b2e636d00ull +
                                   static_cast<std::uint64_t>(stage) + 1u,
                  .source_upper_bytes = source_bytes,
                  .pipeline_stage_count = 1u,
              })) {
        return manifest;
      }
    }
    break;
  }
  case rund::kernel::NodeKind::Gather: {
    manifest = PreparedBackendManifest{.source_build_count = 2u,
                                       .source_library_dependency_count = 2u,
                                       .pipeline_stage_count = 2u,
                                       .descriptor_set_count = 2u,
                                       .descriptor_binding_count = 14u,
                                       .descriptor_lease_count = 2u,
                                       .descriptor_dependency_count = 2u};
    const auto element = step.operation.get<operation::Gather>().plan.element;
    for (const bool control : {true, false}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanGatherSourceBytes(element, control, source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = control ? 0x76756c6b2e676374ull
                                                     : 0x76756c6b2e676174ull,
                            .source_upper_bytes = source_bytes,
                            .pipeline_stage_count = 1u,
                        })) {
        return manifest;
      }
    }
    break;
  }
  case rund::kernel::NodeKind::Histogram: {
    manifest = PreparedBackendManifest{.source_build_count = 2u,
                                       .source_library_dependency_count = 2u,
                                       .pipeline_stage_count = 2u,
                                       .descriptor_set_count = 2u,
                                       .descriptor_binding_count = 8u,
                                       .descriptor_lease_count = 2u,
                                       .descriptor_dependency_count = 2u};
    for (const bool clear : {true, false}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanHistogramSourceBytes(clear, source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = clear ? 0x76756c6b2e68636cull
                                                   : 0x76756c6b2e686374ull,
                            .source_upper_bytes = source_bytes,
                            .pipeline_stage_count = 1u,
                        })) {
        return manifest;
      }
    }
    break;
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
      return manifest;
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
        return manifest;
      }
    }
    const auto add_scan_stage = [&](const VulkanScanStage stage) noexcept {
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
    if (!add_scan_stage(VulkanScanStage::Block) ||
        (scan != 1u && (!add_scan_stage(VulkanScanStage::Prefix) ||
                        !add_scan_stage(VulkanScanStage::Offset)))) {
      return manifest;
    }
    break;
  }
  case rund::kernel::NodeKind::Reduce: {
    const auto &active = step.operation.get<operation::Reduce>();
    const std::uint64_t passes = active.plan.pass_count;
    if (!rund::kernel::checked::mul(6u, passes,
                                    manifest.descriptor_binding_count)) {
      return manifest;
    }
    manifest.source_build_count = 1u;
    manifest.source_library_dependency_count = 1u;
    manifest.pipeline_stage_count = 1u;
    manifest.descriptor_set_count = passes;
    manifest.descriptor_lease_count = passes;
    manifest.descriptor_dependency_count = 1u;
    std::uint64_t source_bytes = 0u;
    if (!VulkanReduceSourceBytes(active.desc.op, active.desc.element,
                                 active.desc.block_size, plan.domain,
                                 source_bytes) ||
        !AddPreparedBackendCacheDependency(
            manifest, PreparedBackendCacheDependency{
                          .source_recipe = 0x76756c6b2e726564ull,
                          .source_upper_bytes = source_bytes,
                          .pipeline_stage_count = 1u,
                      })) {
      return manifest;
    }
    break;
  }
  case rund::kernel::NodeKind::Scatter: {
    manifest = PreparedBackendManifest{.source_build_count = 1u,
                                       .source_library_dependency_count = 1u,
                                       .pipeline_stage_count = 1u,
                                       .descriptor_set_count = 1u,
                                       .descriptor_binding_count = 5u,
                                       .descriptor_lease_count = 1u,
                                       .descriptor_dependency_count = 1u};
    std::uint64_t source_bytes = 0u;
    if (!VulkanScatterSourceBytes(
            step.operation.get<operation::Scatter>().plan.element,
            source_bytes) ||
        !AddPreparedBackendCacheDependency(
            manifest, PreparedBackendCacheDependency{
                          .source_recipe = 0x76756c6b2e736361ull,
                          .source_upper_bytes = source_bytes,
                          .pipeline_stage_count = 1u,
                      })) {
      return manifest;
    }
    break;
  }
  case rund::kernel::NodeKind::ScatterReduce: {
    manifest = PreparedBackendManifest{.source_build_count = 3u,
                                       .source_library_dependency_count = 3u,
                                       .pipeline_stage_count = 3u,
                                       .descriptor_set_count = 3u,
                                       .descriptor_binding_count = 24u,
                                       .descriptor_lease_count = 3u,
                                       .descriptor_dependency_count = 3u};
    const auto &active = step.operation.get<operation::ScatterReduce>();
    for (const VulkanScatterReduceStage stage :
         {VulkanScatterReduceStage::Control, VulkanScatterReduceStage::Init,
          VulkanScatterReduceStage::Fold}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanScatterReduceSourceBytes(active.plan, stage, source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = 0x76756c6b2e737200ull +
                                             static_cast<std::uint64_t>(stage),
                            .source_upper_bytes = source_bytes,
                            .pipeline_stage_count = 1u,
                        })) {
        return manifest;
      }
    }
    break;
  }
  case rund::kernel::NodeKind::Stencil:
  case rund::kernel::NodeKind::Window: {
    const RangePlan &range = *RangePlanFor(step.operation);
    const std::optional<RangeExec> execution = RangeExec::from(range);
    if (!execution.has_value()) {
      return manifest;
    }
    const std::uint64_t stage_count = range.stage_count();
    const bool controlled = step.kind() == rund::kernel::NodeKind::Window &&
                            range.shape().resident_counted();
    const std::uint32_t descriptor_count = execution->descriptor_count();
    std::uint64_t descriptor_bindings = 0u;
    std::uint64_t pipeline_stages = 0u;
    std::uint64_t descriptor_sets = 0u;
    if (!range.ok() || stage_count == 0u ||
        !rund::kernel::checked::mul(stage_count, descriptor_count,
                                    descriptor_bindings) ||
        !rund::kernel::checked::add(stage_count, controlled ? 1u : 0u,
                                    pipeline_stages) ||
        !rund::kernel::checked::add(stage_count, controlled ? 1u : 0u,
                                    descriptor_sets) ||
        (controlled && !rund::kernel::checked::add(descriptor_bindings, 4u,
                                                   descriptor_bindings))) {
      return manifest;
    }
    manifest = PreparedBackendManifest{
        .source_build_count = controlled ? 2u : 1u,
        .source_library_dependency_count = controlled ? 2u : 1u,
        .pipeline_stage_count = pipeline_stages,
        .descriptor_set_count = descriptor_sets,
        .descriptor_binding_count = descriptor_bindings,
        .descriptor_lease_count = descriptor_sets,
        .descriptor_dependency_count = pipeline_stages};
    std::uint64_t source_bytes = 0u;
    if (!VulkanRangeSourceBytes(*execution, source_bytes) ||
        !AddPreparedBackendCacheDependency(
            manifest, PreparedBackendCacheDependency{
                          .source_recipe = 0x76756c6b2e726e67ull,
                          .source_upper_bytes = source_bytes,
                          .pipeline_stage_count = 1u,
                      })) {
      return manifest;
    }
    if (controlled) {
      std::uint64_t control_source_bytes = 0u;
      if (!VulkanRangeControlSourceBytes(range, control_source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = 0x76756c6b2e726374ull,
                            .source_upper_bytes = control_source_bytes,
                            .pipeline_stage_count = 1u,
                        })) {
        return manifest;
      }
    }
    break;
  }
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
  case rund::kernel::NodeKind::Factor:
  case rund::kernel::NodeKind::Solve:
  case rund::kernel::NodeKind::Spectrum: {
    const std::uint64_t bindings =
        step.kind() == rund::kernel::NodeKind::Matrix
            ? 4u
            : (step.kind() == rund::kernel::NodeKind::Factor ||
                       step.kind() == rund::kernel::NodeKind::Spectrum
                   ? 5u
                   : 6u);
    manifest = PreparedBackendManifest{.source_build_count = 1u,
                                       .source_library_dependency_count = 1u,
                                       .pipeline_stage_count = 1u,
                                       .descriptor_set_count = 1u,
                                       .descriptor_binding_count = bindings,
                                       .descriptor_lease_count = 1u,
                                       .descriptor_dependency_count = 1u};
    bool wide = false;
    switch (step.kind()) {
    case rund::kernel::NodeKind::Transform:
      wide = step.operation.get<operation::Transform>().plan.element_bytes ==
             sizeof(rund::kernel::u64);
      break;
    case rund::kernel::NodeKind::Matrix:
      wide = step.operation.get<operation::Matrix>().plan.element_bytes ==
             sizeof(rund::kernel::u64);
      break;
    case rund::kernel::NodeKind::Factor:
      wide = step.operation.get<operation::Factor>().plan.element_bytes ==
             sizeof(rund::kernel::u64);
      break;
    case rund::kernel::NodeKind::Solve:
      wide = step.operation.get<operation::Solve>().plan.element_bytes ==
             sizeof(rund::kernel::u64);
      break;
    case rund::kernel::NodeKind::Spectrum:
      wide = step.operation.get<operation::Spectrum>().plan.element_bytes ==
             sizeof(rund::kernel::u64);
      break;
    default:
      return manifest;
    }
    std::uint64_t source_bytes = 0u;
    if (!VulkanNumericSourceBytes(step.kind(), wide, source_bytes) ||
        !AddPreparedBackendCacheDependency(
            manifest,
            PreparedBackendCacheDependency{
                .source_recipe = 0x76756c6b2e6e7500ull +
                                 static_cast<std::uint64_t>(step.kind()),
                .source_upper_bytes = source_bytes,
                .pipeline_stage_count = 1u,
            })) {
      return manifest;
    }
    break;
  }
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
