#include "../../../context/internal/support.hpp"
#include "../../../kernel/backend/execute.hpp"
#include "../../../kernel/backend/template_plan.hpp"
#include "../../../kernel/status.hpp"

#include "../../../sort/block/metal.hpp"
#include "../../buffer/owner.hpp"
#include "../../compact/local.hpp"
#include "../../gather/local.hpp"
#include "../../histogram/local.hpp"
#include "../../kernel.hpp"
#include "../../numeric/source.hpp"
#include "../../numeric/state.hpp"
#include "../../partition/local.hpp"
#include "../../pipeline/guard.hpp"
#include "../../pipeline/source_recipe.hpp"
#include "../../range/local.hpp"
#include "../../reduce/local.hpp"
#include "../../runtime/map/source_upper.hpp"
#include "../../scan/local.hpp"
#include "../../scan/source.hpp"
#include "../../scatter/local.hpp"
#include "../../scatter/reduce/model.hpp"
#include "../../segmented/local.hpp"
#include "../../segmented/reduce/model.hpp"
#include "../../sort/source.hpp"
#include "../manifest.hpp"
#include "../ops/prepare.hpp"
#include "../pipeline/build.hpp"
#include "../pipeline/identity_index.hpp"
#include "parameter.hpp"
#include "source_recipe.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] bool
AddMetalPipelineSourceRecipe(PreparedBackendManifest &manifest,
                             const MetalPipelineSourceRecipe recipe) noexcept {
  std::uint64_t raw_storage_upper = 0u;
  if (!recipe.ok || !backend_source_recipe::string_external_storage_upper_bytes(
                        recipe.raw_source_upper_bytes, raw_storage_upper)) {
    return false;
  }
  manifest.cold_source_transient_bytes =
      std::max(manifest.cold_source_transient_bytes, raw_storage_upper);
  return AddPreparedBackendCacheDependency(
      manifest, PreparedBackendCacheDependency{
                    .source_recipe = recipe.recipe_id,
                    .source_upper_bytes = recipe.final_source_upper_bytes,
                    .pipeline_stage_count = recipe.pipeline_stage_count,
                });
}

[[nodiscard]] MetalPipelineSourceRecipe MetalScanPipelineSourceRecipe(
    const std::uint64_t pipeline_count = 3u) noexcept {
  if (pipeline_count != 1u && pipeline_count != 3u) {
    return {};
  }
  std::uint64_t raw_upper = 0u;
  return MetalScanSourceUpperBytes(raw_upper)
             ? MetalSourceRecipe(0x6d6574616c736361ull, raw_upper, 7u,
                                 pipeline_count)
             : MetalPipelineSourceRecipe{};
}

[[nodiscard]] MetalPipelineSourceRecipe
MetalSegmentedScanPipelineSourceRecipe() noexcept {
  std::uint64_t raw_upper = 0u;
  return MetalSegmentedScanSourceUpperBytes(raw_upper)
             ? MetalSourceRecipe(0x6d6574616c736567ull, raw_upper, 3u, 3u)
             : MetalPipelineSourceRecipe{};
}

[[nodiscard]] MetalPipelineSourceRecipe
MetalSortPipelineSourceRecipe() noexcept {
  std::uint64_t raw_upper = 0u;
  return MetalSortSourceUpperBytes(kMetalSortBlockSize, raw_upper)
             ? MetalSourceRecipe(0x6d6574616c736f72ull, raw_upper, 7u, 5u)
             : MetalPipelineSourceRecipe{};
}

[[nodiscard]] MetalPipelineSourceRecipe
MetalCompactPipelineSourceRecipe() noexcept {
  std::uint64_t raw_upper = 0u;
  return MetalCompactSourceUpperBytes(raw_upper)
             ? MetalSourceRecipe(0x6d6574616c636f6dull, raw_upper, 4u, 2u)
             : MetalPipelineSourceRecipe{};
}

[[nodiscard]] MetalPipelineSourceRecipe
MetalGatherPipelineSourceRecipe() noexcept {
  return MetalSourceRecipe(0x6d6574616c676174ull, MetalGatherSourceUpperBytes(),
                           3u, 2u);
}

[[nodiscard]] MetalPipelineSourceRecipe
MetalHistogramPipelineSourceRecipe() noexcept {
  return MetalSourceRecipe(0x6d6574616c686973ull,
                           MetalHistogramSourceUpperBytes(), 2u, 2u);
}

[[nodiscard]] MetalPipelineSourceRecipe
MetalPartitionPipelineSourceRecipe() noexcept {
  std::uint64_t raw_upper = 0u;
  return MetalPartitionSourceUpperBytes(raw_upper)
             ? MetalSourceRecipe(0x6d6574616c706172ull, raw_upper, 6u, 2u)
             : MetalPipelineSourceRecipe{};
}

[[nodiscard]] MetalPipelineSourceRecipe
MetalScatterPipelineSourceRecipe() noexcept {
  return MetalSourceRecipe(0x6d6574616c736374ull,
                           MetalScatterSourceUpperBytes(), 2u, 1u);
}

[[nodiscard]] MetalPipelineSourceRecipe
MetalSegmentedReducePipelineSourceRecipe(
    const rund::kernel::SegmentedReducePlan &plan,
    const rund::kernel::ComputeDomain domain) noexcept {
  std::uint64_t raw_upper = 0u;
  return MetalSegmentedReduceSourceUpperBytes(plan.op, domain, raw_upper)
             ? MetalSourceRecipe(0x6d6574616c737264ull, raw_upper, 5u, 4u)
             : MetalPipelineSourceRecipe{};
}

[[nodiscard]] MetalPipelineSourceRecipe MetalReducePipelineSourceRecipe(
    const rund::kernel::ReducePlan &plan,
    const rund::kernel::ComputeDomain domain) noexcept {
  std::uint64_t raw_upper = 0u;
  return MetalReduceSourceUpperBytes(plan.op, plan.block_size, domain,
                                     raw_upper)
             ? MetalSourceRecipe(0x6d6574616c726564ull, raw_upper, 2u, 1u)
             : MetalPipelineSourceRecipe{};
}

[[nodiscard]] MetalPipelineSourceRecipe MetalScatterReducePipelineSourceRecipe(
    const rund::kernel::ScatterReducePlan &plan) noexcept {
  std::uint64_t raw_upper = 0u;
  return MetalScatterReduceSourceUpperBytes(plan, raw_upper)
             ? MetalSourceRecipe(0x6d6574616c736372ull, raw_upper, 3u, 3u)
             : MetalPipelineSourceRecipe{};
}

[[nodiscard]] MetalPipelineSourceRecipe
MetalRangeSourceRecipe(const RangePlan &range) noexcept {
  const std::optional<RangeExec> execution = RangeExec::from(range);
  std::uint64_t raw_upper = 0u;
  return execution.has_value() &&
                 MetalRangeSourceUpperBytes(*execution, raw_upper)
             ? MetalSourceRecipe(0x6d6574616c726e67ull, raw_upper, 4u,
                                 range.stage_count())
             : MetalPipelineSourceRecipe{};
}

[[nodiscard]] MetalPipelineSourceRecipe
MetalRangeControlSourceRecipe(const RangePlan &range) noexcept {
  std::uint64_t raw_upper = 0u;
  return range.ok() && range.shape().resident_counted() &&
                 MetalRangeControlSourceUpperBytes(range, raw_upper)
             ? MetalSourceRecipe(0x6d6574616c726374ull, raw_upper, 1u, 1u)
             : MetalPipelineSourceRecipe{};
}

[[nodiscard]] MetalPipelineSourceRecipe
MetalNumericPipelineSourceRecipe() noexcept {
  std::uint64_t raw_upper = 0u;
  return MetalNumericSourceUpperBytes(raw_upper)
             ? MetalSourceRecipe(0x6d6574616c6e756dull, raw_upper, 10u, 1u)
             : MetalPipelineSourceRecipe{};
}

} // namespace

bool AddMetalStepSourceRecipes(const KernelExecutionStep &step,
                               const rund::kernel::ComputePlan &plan,
                               const bool controlled, const bool has_checks,
                               PreparedBackendManifest &manifest) noexcept {
  const auto dimensions = [&](const std::uint64_t source_builds,
                              const std::uint64_t stages,
                              const std::uint64_t source_libraries) {
    manifest.source_build_count = source_builds;
    manifest.pipeline_stage_count = stages;
    manifest.source_library_dependency_count = source_libraries;
  };
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map: {
    const std::uint64_t control_stage = controlled ? 1u : 0u;
    const std::uint64_t check_stage = has_checks ? 1u : 0u;
    dimensions(1u + control_stage + check_stage,
               1u + control_stage + check_stage,
               1u + control_stage + check_stage);
    std::uint64_t main_source = 0u;
    std::uint64_t ignored_transient = 0u;
    std::uint64_t controlled_source = 0u;
    std::uint64_t guarded_source = 0u;
    if (!backend_template_plan::map_source_upper(step, plan, main_source,
                                                 ignored_transient) ||
        (controlled && !MetalControlledMapSourceUpperBytes(
                           plan, main_source, controlled_source)) ||
        !PipelinePrivateMetalSourceUpperBytes(controlled ? controlled_source
                                                         : main_source,
                                              1u, true, guarded_source) ||
        !AddPreparedBackendCacheDependency(
            manifest, PreparedBackendCacheDependency{
                          .source_recipe = 0x6d6574616c2e6d61ull,
                          .source_upper_bytes = guarded_source,
                          .pipeline_stage_count = 1u,
                      })) {
      return false;
    }
    if (controlled) {
      std::uint64_t control_source = 0u;
      if (!PipelinePrivateMetalSourceUpperBytes(
              MetalMapControlSourceText().size(), 1u, true, control_source) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = 0x6d6574616c2e6374ull,
                            .source_upper_bytes = control_source,
                            .pipeline_stage_count = 1u,
                        })) {
        return false;
      }
    }
    if (has_checks) {
      std::uint64_t check_source = 0u;
      if (!MetalMapCheckSourceUpperBytes(step.artifact, check_source) ||
          !PipelinePrivateMetalSourceUpperBytes(check_source, 1u, true,
                                                guarded_source) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = 0x6d6574616c2e6368ull,
                            .source_upper_bytes = guarded_source,
                            .pipeline_stage_count = 1u,
                        })) {
        return false;
      }
    }
    break;
  }
  case rund::kernel::NodeKind::Scan: {
    const RangePrefixExec prefix_execution =
        PlanScanPrefixExecution(step.operation.get<operation::Scan>().plan);
    const std::uint64_t pipeline_count =
        MetalScanPipelineCount(prefix_execution);
    if (pipeline_count == 0u) {
      return false;
    }
    dimensions(1u, pipeline_count, 1u);
    if (!AddMetalPipelineSourceRecipe(
            manifest, MetalScanPipelineSourceRecipe(pipeline_count))) {
      return false;
    }
    break;
  }
  case rund::kernel::NodeKind::SegmentedScan:
    dimensions(1u, 3u, 1u);
    if (!AddMetalPipelineSourceRecipe(
            manifest, MetalSegmentedScanPipelineSourceRecipe())) {
      return false;
    }
    break;
  case rund::kernel::NodeKind::SegmentedReduce:
    dimensions(1u, 4u, 1u);
    if (!AddMetalPipelineSourceRecipe(
            manifest, MetalSegmentedReducePipelineSourceRecipe(
                          step.operation.get<operation::SegmentedReduce>().plan,
                          plan.domain))) {
      return false;
    }
    break;
  case rund::kernel::NodeKind::Sort:
    dimensions(1u, 5u, 1u);
    if (!AddMetalPipelineSourceRecipe(manifest,
                                      MetalSortPipelineSourceRecipe())) {
      return false;
    }
    break;
  case rund::kernel::NodeKind::Compact:
    dimensions(2u, 5u, 2u);
    if (!AddMetalPipelineSourceRecipe(manifest,
                                      MetalCompactPipelineSourceRecipe()) ||
        !AddMetalPipelineSourceRecipe(manifest,
                                      MetalScanPipelineSourceRecipe())) {
      return false;
    }
    break;
  case rund::kernel::NodeKind::Gather:
    dimensions(1u, 2u, 1u);
    if (!AddMetalPipelineSourceRecipe(manifest,
                                      MetalGatherPipelineSourceRecipe())) {
      return false;
    }
    break;
  case rund::kernel::NodeKind::Histogram:
    dimensions(1u, 2u, 1u);
    if (!AddMetalPipelineSourceRecipe(manifest,
                                      MetalHistogramPipelineSourceRecipe())) {
      return false;
    }
    break;
  case rund::kernel::NodeKind::Partition: {
    const RangePrefixExec scan_execution = MetalPartitionScanExecution(
        step.operation.get<operation::Partition>().plan);
    const std::uint64_t scan_pipeline_count =
        MetalScanPipelineCount(scan_execution);
    const std::uint64_t pipeline_count =
        MetalPartitionPipelineCount(scan_execution);
    if (pipeline_count == 0u) {
      return false;
    }
    dimensions(3u, pipeline_count, 2u);
    if (!AddMetalPipelineSourceRecipe(manifest,
                                      MetalPartitionPipelineSourceRecipe()) ||
        !AddMetalPipelineSourceRecipe(
            manifest, MetalScanPipelineSourceRecipe(scan_pipeline_count))) {
      return false;
    }
    break;
  }
  case rund::kernel::NodeKind::Reduce:
    dimensions(1u, 1u, 1u);
    if (!AddMetalPipelineSourceRecipe(
            manifest,
            MetalReducePipelineSourceRecipe(
                step.operation.get<operation::Reduce>().plan, plan.domain))) {
      return false;
    }
    break;
  case rund::kernel::NodeKind::Scatter:
    dimensions(1u, 1u, 1u);
    if (!AddMetalPipelineSourceRecipe(manifest,
                                      MetalScatterPipelineSourceRecipe())) {
      return false;
    }
    break;
  case rund::kernel::NodeKind::Stencil:
  case rund::kernel::NodeKind::Window: {
    const RangePlan &range = *RangePlanFor(step.operation);
    const bool resident = range.shape().resident_counted();
    dimensions(resident ? 2u : 1u, range.stage_count() + (resident ? 1u : 0u),
               resident ? 2u : 1u);
    if (!AddMetalPipelineSourceRecipe(manifest,
                                      MetalRangeSourceRecipe(range)) ||
        (resident && !AddMetalPipelineSourceRecipe(
                         manifest, MetalRangeControlSourceRecipe(range)))) {
      return false;
    }
    break;
  }
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
  case rund::kernel::NodeKind::Factor:
  case rund::kernel::NodeKind::Solve:
  case rund::kernel::NodeKind::Spectrum:
    dimensions(1u, 1u, 1u);
    if (!AddMetalPipelineSourceRecipe(manifest,
                                      MetalNumericPipelineSourceRecipe())) {
      return false;
    }
    break;
  case rund::kernel::NodeKind::ScatterReduce:
    dimensions(1u, 3u, 1u);
    if (!AddMetalPipelineSourceRecipe(
            manifest,
            MetalScatterReducePipelineSourceRecipe(
                step.operation.get<operation::ScatterReduce>().plan))) {
      return false;
    }
    break;
  }
  return true;
}
#endif

} // namespace rund::node::accel::detail
