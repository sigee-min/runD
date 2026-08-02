#include "../../context/internal/support.hpp"
#include "../../kernel/backend/execute.hpp"
#include "../../kernel/backend/template_plan.hpp"
#include "../../kernel/status.hpp"

#include "../../sort/block/metal.hpp"
#include "../buffer/owner.hpp"
#include "../compact/local.hpp"
#include "../gather/local.hpp"
#include "../histogram/local.hpp"
#include "../kernel.hpp"
#include "../numeric/source.hpp"
#include "../numeric/state.hpp"
#include "../partition/local.hpp"
#include "../pipeline/guard.hpp"
#include "../pipeline/source_recipe.hpp"
#include "../reduce/local.hpp"
#include "../runtime/map/source_upper.hpp"
#include "../scan/source.hpp"
#include "../scatter/local.hpp"
#include "../scatter/reduce/model.hpp"
#include "../segmented/local.hpp"
#include "../segmented/reduce/model.hpp"
#include "../sort/source.hpp"
#include "../stencil/local.hpp"
#include "manifest.hpp"
#include "ops/prepare.hpp"
#include "pipeline/build.hpp"
#include "pipeline/identity_index.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] bool AddMetalHostBytes(std::uint64_t &target,
                                     const std::uint64_t count,
                                     const std::uint64_t element) noexcept {
  std::uint64_t bytes = 0u;
  return backend_template_plan::product(count, element, bytes) &&
         backend_template_plan::add(target, bytes);
}

[[nodiscard]] bool
AddAlignedMetalParameterBytes(std::uint64_t &target,
                              const std::uint64_t bytes) noexcept {
  constexpr std::uint64_t Alignment = 16u;
  std::uint64_t padded = bytes;
  return bytes != 0u && backend_template_plan::add(padded, Alignment - 1u) &&
         (padded &= ~(Alignment - 1u), true) &&
         backend_template_plan::add(target, padded);
}

// Producer-adjacent Metal argument authority. Values are the highest
// non-guard index authored by an encoder plus one; they are prefix uppers, not
// counts to add across stages. The capture binding mask persists between
// commands, so the stream-wide safe cardinality is the maximum active prefix.
[[nodiscard]] bool PlanMetalCaptureBindingSlotUpper(
    const KernelExecutionStep &step, const rund::kernel::ComputePlan &plan,
    const bool controlled, const bool has_checks, std::uint64_t &out) noexcept {
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map: {
    std::uint64_t main = plan.input_buffer_count;
    std::uint64_t checks = MetalMapUniqueCheckCount(step.artifact);
    if (!backend_template_plan::add(main, plan.output_buffer_count) ||
        !backend_template_plan::add(main, 2u) ||
        (has_checks && !backend_template_plan::add(checks, 4u))) {
      return false;
    }
    out = std::max(main, controlled ? std::uint64_t{6u} : 0u);
    if (has_checks) {
      out = std::max(out, checks);
    }
    break;
  }
  case rund::kernel::NodeKind::Scan:
    out = 11u;
    break;
  case rund::kernel::NodeKind::SegmentedScan:
    out = 7u;
    break;
  case rund::kernel::NodeKind::SegmentedReduce:
    out = 6u;
    break;
  case rund::kernel::NodeKind::Sort:
    out = 8u;
    break;
  case rund::kernel::NodeKind::Compact:
    // Compact embeds the full Scan producer whose block stage authors 0...10.
    out = 11u;
    break;
  case rund::kernel::NodeKind::Gather:
    out = 5u;
    break;
  case rund::kernel::NodeKind::Histogram:
    out = 4u;
    break;
  case rund::kernel::NodeKind::Partition:
    // Partition embeds that same full Scan producer before scatter.
    out = 11u;
    break;
  case rund::kernel::NodeKind::Reduce:
    out = 6u;
    break;
  case rund::kernel::NodeKind::Scatter:
    out = 5u;
    break;
  case rund::kernel::NodeKind::ScatterReduce:
    out = 8u;
    break;
  case rund::kernel::NodeKind::Stencil:
    out = 3u;
    break;
  case rund::kernel::NodeKind::Transform:
    out = 6u;
    break;
  case rund::kernel::NodeKind::Matrix:
    out = 4u;
    break;
  case rund::kernel::NodeKind::Factor:
    out = 5u;
    break;
  case rund::kernel::NodeKind::Solve:
    out = 6u;
    break;
  case rund::kernel::NodeKind::Spectrum:
    out = 5u;
    break;
  }
  return out != 0u && out <= kMetalPipelineGuardBinding;
}

struct MetalStepControlShape final {
  std::uint64_t status_source_count{};
  std::uint64_t status_entry_count{};
  std::uint64_t status_command_count{};
  std::uint64_t status_parameter_bytes{};
  std::uint64_t telemetry_source_count{};
  bool ok{};
};

[[nodiscard]] MetalStepControlShape
PlanMetalStepControlShape(const KernelExecutionStep &step,
                          const BoundStep *const bound) noexcept {
  MetalStepControlShape result{};
  const rund::kernel::GraphControl &control =
      bound == nullptr ? step.control : bound->control.control;
  const bool active_control =
      bound == nullptr
          ? step.control.has_count() || step.control.has_predicate()
          : bound->control.active();
  const auto status = [&](const std::uint64_t sources,
                          const std::uint64_t entries,
                          const std::uint64_t imports) noexcept {
    if (sources == 0u || entries == 0u || imports > sources) {
      return false;
    }
    result.status_source_count = sources;
    result.status_entry_count = entries;
    result.status_command_count = imports;
    if (!backend_template_plan::add(result.status_command_count, 1u)) {
      return false;
    }
    std::uint64_t bytes = 0u;
    if (!backend_template_plan::product(
            entries, sizeof(MetalPipelineStatusEntryMeta), bytes) ||
        !AddAlignedMetalParameterBytes(result.status_parameter_bytes, bytes) ||
        !backend_template_plan::product(
            sources, sizeof(MetalPipelineStatusSourceMeta), bytes) ||
        !AddAlignedMetalParameterBytes(result.status_parameter_bytes, bytes) ||
        !AddAlignedMetalParameterBytes(result.status_parameter_bytes,
                                       sizeof(MetalPipelineStatusParams))) {
      return false;
    }
    for (std::uint64_t index = 0u; index < imports; ++index) {
      if (!AddAlignedMetalParameterBytes(result.status_parameter_bytes,
                                         2u * sizeof(std::uint32_t))) {
        return false;
      }
    }
    return true;
  };

  bool valid = true;
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map: {
    const bool described =
        active_control || !step.artifact.metadata.read_routes.empty();
    valid = !described || status(1u, 1u, 0u);
    result.telemetry_source_count = described ? 1u : 0u;
    break;
  }
  case rund::kernel::NodeKind::Scan:
    valid = status(1u, 1u, 0u);
    result.telemetry_source_count =
        control.iteration != 0u && control.has_count() ? 1u : 0u;
    break;
  case rund::kernel::NodeKind::SegmentedScan:
  case rund::kernel::NodeKind::SegmentedReduce:
  case rund::kernel::NodeKind::Gather:
  case rund::kernel::NodeKind::Histogram:
  case rund::kernel::NodeKind::Partition:
  case rund::kernel::NodeKind::Reduce:
  case rund::kernel::NodeKind::Scatter:
  case rund::kernel::NodeKind::ScatterReduce:
    valid = status(1u, 1u, 0u);
    break;
  case rund::kernel::NodeKind::Sort: {
    const bool described =
        step.operation.get<operation::Sort>().plan.count_source !=
        rund::kernel::ComputeCountSource::Descriptor;
    valid = !described || status(1u, 1u, 0u);
    result.telemetry_source_count =
        control.iteration != 0u && control.has_count() ? 1u : 0u;
    break;
  }
  case rund::kernel::NodeKind::Compact: {
    const std::uint64_t sources =
        step.operation.get<operation::Compact>().plan.status_bytes == 0u ? 1u
                                                                         : 2u;
    valid = status(sources, sources, 0u);
    break;
  }
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
  case rund::kernel::NodeKind::Stencil:
    break;
  case rund::kernel::NodeKind::Factor:
    valid = status(
        1u, step.operation.get<operation::Factor>().plan.status_count, 1u);
    break;
  case rund::kernel::NodeKind::Solve:
    valid = status(1u, step.operation.get<operation::Solve>().plan.status_count,
                   1u);
    break;
  case rund::kernel::NodeKind::Spectrum:
    valid = status(
        1u, step.operation.get<operation::Spectrum>().plan.status_count, 1u);
    break;
  }
  result.ok = valid;
  return result;
}

[[nodiscard]] bool
AddMetalMapTemplateHostBytes(std::uint64_t &target,
                             const rund::kernel::ComputePlan &plan,
                             const std::uint64_t check_count) noexcept {
  return backend_template_plan::add(target,
                                    sizeof(MetalMapTemplateResources)) &&
         AddMetalHostBytes(target, plan.input_buffer_count,
                           sizeof(InputWindowPlan)) &&
         AddMetalHostBytes(target, plan.input_buffer_count,
                           sizeof(std::uint64_t)) &&
         AddMetalHostBytes(target, plan.output_buffer_count,
                           sizeof(std::uint64_t)) &&
         AddMetalHostBytes(target, check_count, sizeof(MetalMapCheck));
}

[[nodiscard]] bool
AddMetalMapRouteHostBytes(std::uint64_t &target,
                          const rund::kernel::ComputePlan &plan) noexcept {
  return backend_template_plan::add(target, sizeof(MetalMapEncodeResources)) &&
         AddMetalHostBytes(target,
                           plan.input_buffer_count > kInlineMetalBufferCount
                               ? plan.input_buffer_count -
                                     kInlineMetalBufferCount
                               : 0u,
                           sizeof(MetalResidentBufferResult)) &&
         AddMetalHostBytes(target,
                           plan.output_buffer_count > kInlineMetalBufferCount
                               ? plan.output_buffer_count -
                                     kInlineMetalBufferCount
                               : 0u,
                           sizeof(MetalResidentBufferResult));
}

struct MetalRecurrenceSourceReservation final {
  std::uint64_t source_bytes{};
  std::uint64_t source_storage_bytes{};
  std::uint64_t transient_bytes{};
  std::uint64_t native_object_count{};
  bool ok{};
};

[[nodiscard]] MetalRecurrenceSourceReservation
PlanMetalRecurrenceSource(const MapRecurrenceSourcePlan &source,
                          const rund::kernel::ComputePlan &plan,
                          const bool history) noexcept {
  MetalRecurrenceSourceReservation result{};
  std::uint64_t specialized_upper = 0u;
  if (!source.ok || source.history != history ||
      source.exact_source_bytes == 0u ||
      source.source_upper_bytes < source.exact_source_bytes ||
      source.source_storage_upper_bytes == 0u ||
      !MapSpecializedSourceUpperBytes(source.exact_source_bytes,
                                      source.source_upper_bytes, plan,
                                      specialized_upper) ||
      !PipelinePrivateMetalSourceUpperBytes(specialized_upper, 1u, true,
                                            result.source_bytes) ||
      !backend_source_recipe::string_external_storage_upper_bytes(
          result.source_bytes, result.source_storage_bytes)) {
    return result;
  }
  // The one source allocation moves into the adapter cache and is retained
  // template storage. Only the semantic metadata that is discarded after
  // in-place specialization is transient.
  result.transient_bytes = source.metadata_storage_upper_bytes;
  // One source compilation requests one MTLLibrary and one pipeline state.
  // Their opaque byte sizes are not guessed; the structural object count is
  // the auditable native reservation.
  result.native_object_count = 2u;
  result.ok = true;
  return result;
}

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

[[nodiscard]] MetalPipelineSourceRecipe
MetalScanPipelineSourceRecipe() noexcept {
  std::uint64_t raw_upper = 0u;
  return MetalScanSourceUpperBytes(raw_upper)
             ? MetalSourceRecipe(0x6d6574616c736361ull, raw_upper, 7u, 3u)
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

[[nodiscard]] MetalPipelineSourceRecipe MetalStencilPipelineSourceRecipe(
    const rund::kernel::StencilPlan &plan) noexcept {
  std::uint64_t raw_upper = 0u;
  return MetalStencilSourceUpperBytes(plan.op, raw_upper)
             ? MetalSourceRecipe(0x6d6574616c73746eull, raw_upper, 4u, 1u)
             : MetalPipelineSourceRecipe{};
}

[[nodiscard]] MetalPipelineSourceRecipe
MetalNumericPipelineSourceRecipe() noexcept {
  std::uint64_t raw_upper = 0u;
  return MetalNumericSourceUpperBytes(raw_upper)
             ? MetalSourceRecipe(0x6d6574616c6e756dull, raw_upper, 10u, 1u)
             : MetalPipelineSourceRecipe{};
}

[[nodiscard]] bool AddMetalViewDispatchCount(
    const KernelExecutionStep &step, const KernelViewLayout *const views,
    PreparedKernelRouteReservation &reservation) noexcept {
  if (views == nullptr || step.kind() == rund::kernel::NodeKind::Map ||
      step.kind() == rund::kernel::NodeKind::ScatterReduce) {
    return true;
  }
  std::uint64_t count = 0u;
  for (std::size_t local = 0u; local < step.graph_binding_indices.size();
       ++local) {
    const std::uint64_t binding = step.graph_binding_indices[local];
    bool first = true;
    for (std::size_t prior = 0u; prior < local; ++prior) {
      if (step.graph_binding_indices[prior] == binding) {
        first = false;
        break;
      }
    }
    if (!first) {
      continue;
    }
    if (std::any_of(views->begin(), views->end(),
                    [binding](const KernelViewSlot &view) {
                      return view.binding == binding;
                    }) &&
        !backend_template_plan::add(count, 1u)) {
      return false;
    }
  }
  if (count != 0u) {
    reservation.capture_binding_slot_upper =
        std::max(reservation.capture_binding_slot_upper, std::uint64_t{3u});
  }
  return backend_template_plan::add(reservation.dispatch_count, count);
}

[[nodiscard]] bool AddMetalPrimitiveDispatchUpper(
    const rund::kernel::ComputePlan &plan,
    const PreparedBackendManifest &manifest,
    PreparedKernelRouteReservation &reservation) noexcept {
  if (plan.dispatch_count == 0u || manifest.pipeline_stage_count == 0u) {
    return false;
  }
  std::uint64_t additional = 0u;
  return backend_template_plan::product(plan.dispatch_count,
                                        manifest.pipeline_stage_count - 1u,
                                        additional) &&
         backend_template_plan::add(reservation.dispatch_count, additional);
}

[[nodiscard]] rund::AccelCheck PlanMetalStepStructure(
    const KernelExecutionStep &step, const rund::kernel::ComputePlan &plan,
    const BoundStep *const bound, const KernelViewLayout *const views,
    const std::uint64_t, PreparedKernelRouteReservation &reservation) noexcept {
  const PreparedBackendManifest manifest =
      BuildMetalBackendManifest(step, plan, bound, 1u);
  if (!manifest.ok) {
    return rund::AccelCheck{false, manifest.reason};
  }
  reservation.capture_binding_slot_upper =
      std::max(reservation.capture_binding_slot_upper,
               manifest.capture_binding_slot_upper);
  if (!AddMetalPrimitiveDispatchUpper(plan, manifest, reservation) ||
      !AddMetalViewDispatchCount(step, views, reservation)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t route = 0u;
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map: {
    std::uint64_t ignored_source_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
    if (!AddMetalMapTemplateHostBytes(
            reservation.template_host_bytes, plan,
            MetalMapUniqueCheckCount(step.artifact)) ||
        !AddMetalMapRouteHostBytes(route, plan) ||
        !backend_template_plan::map_source_upper(
            step, plan, ignored_source_bytes, transient_bytes)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    reservation.source_transient_bytes =
        std::max(reservation.source_transient_bytes, transient_bytes);
    break;
  }
  case rund::kernel::NodeKind::Scan:
    route = sizeof(MetalScanEncodeResources);
    break;
  case rund::kernel::NodeKind::SegmentedScan:
    route = sizeof(MetalSegmentedScanEncodeResources);
    break;
  case rund::kernel::NodeKind::SegmentedReduce:
    route = sizeof(MetalSegmentedReduceResources);
    break;
  case rund::kernel::NodeKind::Sort:
    route = sizeof(MetalSortEncodeResources);
    break;
  case rund::kernel::NodeKind::Compact:
    route = sizeof(MetalCompactEncodeResources);
    break;
  case rund::kernel::NodeKind::Gather:
    route = sizeof(MetalGatherEncodeResources);
    break;
  case rund::kernel::NodeKind::Histogram:
    route = sizeof(MetalHistogramEncodeResources);
    break;
  case rund::kernel::NodeKind::Partition:
    route = sizeof(MetalPartitionEncodeResources);
    break;
  case rund::kernel::NodeKind::Reduce:
    route = sizeof(MetalReduceEncodeResources);
    break;
  case rund::kernel::NodeKind::Scatter:
    route = sizeof(MetalScatterEncodeResources);
    break;
  case rund::kernel::NodeKind::ScatterReduce:
    route = sizeof(MetalScatterReduceResources);
    break;
  case rund::kernel::NodeKind::Stencil:
    route = sizeof(MetalStencilEncodeResources);
    break;
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
  case rund::kernel::NodeKind::Factor:
  case rund::kernel::NodeKind::Solve:
  case rund::kernel::NodeKind::Spectrum:
    route = sizeof(MetalNumericPrepared);
    break;
  }
  const std::uint64_t immutable_host_bytes =
      step.kind() == rund::kernel::NodeKind::Map
          ? 0u
          : sizeof(MetalKernelImmutablePipelines);
  reservation.source_transient_bytes = std::max(
      reservation.source_transient_bytes, manifest.cold_source_transient_bytes);
  return backend_template_plan::add(reservation.route_host_bytes, route) &&
                 backend_template_plan::add(reservation.template_host_bytes,
                                            immutable_host_bytes) &&
                 backend_template_plan::add(reservation.template_source_bytes,
                                            manifest.cold_cache_source_bytes) &&
                 backend_template_plan::add(
                     reservation.template_host_bytes,
                     manifest.cold_cache_source_storage_bytes) &&
                 backend_template_plan::add(
                     reservation.template_native_allocation_count,
                     manifest.cold_cache_native_object_count) &&
                 backend_template_plan::add(reservation.status_source_count,
                                            manifest.status_source_count) &&
                 backend_template_plan::add(reservation.status_entry_count,
                                            manifest.status_entry_count) &&
                 backend_template_plan::add(reservation.status_command_count,
                                            manifest.status_command_count) &&
                 backend_template_plan::add(reservation.status_parameter_bytes,
                                            manifest.status_parameter_bytes) &&
                 backend_template_plan::add(reservation.telemetry_source_count,
                                            manifest.telemetry_source_count)
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "compute_pipeline_capacity"};
}

[[nodiscard]] backend_template_plan::BackendShape MetalBackendShape() noexcept {
  return backend_template_plan::BackendShape{
      .storage_alignment = 1u,
      .max_dispatch_groups = 1u,
      .reset_dispatch_window = std::numeric_limits<std::uint32_t>::max(),
      .template_capacity = PreparedPipelineStepCapacity,
      .route_header_bytes = sizeof(MetalKernelResources),
      .route_step_bytes = sizeof(MetalKernelEntry),
      .route_inline_step_capacity = kInlineBoundStepCapacity,
      .template_header_bytes = sizeof(MetalKernelProgramTemplate),
      .template_step_bytes = sizeof(MetalKernelProgramStepTemplate),
      .plan_step = PlanMetalStepStructure,
  };
}

} // namespace

PreparedBackendManifest BuildMetalBackendManifest(
    const KernelExecutionStep &step, const rund::kernel::ComputePlan &plan,
    const BoundStep *const bound, const std::uint64_t) noexcept {
  PreparedBackendManifest manifest{};
  const bool has_checks = !step.artifact.metadata.read_routes.empty();
  const bool controlled = (bound != nullptr ? bound->control.active()
                                            : (step.control.has_count() ||
                                               step.control.has_predicate())) ||
                          has_checks;
  const auto dimensions = [&](const std::uint64_t source_builds,
                              const std::uint64_t stages,
                              const std::uint64_t source_libraries) {
    manifest.source_build_count = source_builds;
    manifest.pipeline_stage_count = stages;
    manifest.source_library_dependency_count = source_libraries;
  };
  if (!PlanMetalCaptureBindingSlotUpper(step, plan, controlled, has_checks,
                                        manifest.capture_binding_slot_upper)) {
    return manifest;
  }
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
      return manifest;
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
        return manifest;
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
        return manifest;
      }
    }
    break;
  }
  case rund::kernel::NodeKind::Scan:
    dimensions(1u, 3u, 1u);
    if (!AddMetalPipelineSourceRecipe(manifest,
                                      MetalScanPipelineSourceRecipe())) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::SegmentedScan:
    dimensions(1u, 3u, 1u);
    if (!AddMetalPipelineSourceRecipe(
            manifest, MetalSegmentedScanPipelineSourceRecipe())) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::SegmentedReduce:
    dimensions(1u, 4u, 1u);
    if (!AddMetalPipelineSourceRecipe(
            manifest, MetalSegmentedReducePipelineSourceRecipe(
                          step.operation.get<operation::SegmentedReduce>().plan,
                          plan.domain))) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::Sort:
    dimensions(1u, 5u, 1u);
    if (!AddMetalPipelineSourceRecipe(manifest,
                                      MetalSortPipelineSourceRecipe())) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::Compact:
    dimensions(2u, 5u, 2u);
    if (!AddMetalPipelineSourceRecipe(manifest,
                                      MetalCompactPipelineSourceRecipe()) ||
        !AddMetalPipelineSourceRecipe(manifest,
                                      MetalScanPipelineSourceRecipe())) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::Gather:
    dimensions(1u, 2u, 1u);
    if (!AddMetalPipelineSourceRecipe(manifest,
                                      MetalGatherPipelineSourceRecipe())) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::Histogram:
    dimensions(1u, 2u, 1u);
    if (!AddMetalPipelineSourceRecipe(manifest,
                                      MetalHistogramPipelineSourceRecipe())) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::Partition:
    dimensions(3u, 5u, 2u);
    if (!AddMetalPipelineSourceRecipe(manifest,
                                      MetalPartitionPipelineSourceRecipe()) ||
        !AddMetalPipelineSourceRecipe(manifest,
                                      MetalScanPipelineSourceRecipe())) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::Reduce:
    dimensions(1u, 1u, 1u);
    if (!AddMetalPipelineSourceRecipe(
            manifest,
            MetalReducePipelineSourceRecipe(
                step.operation.get<operation::Reduce>().plan, plan.domain))) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::Scatter:
    dimensions(1u, 1u, 1u);
    if (!AddMetalPipelineSourceRecipe(manifest,
                                      MetalScatterPipelineSourceRecipe())) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::Stencil:
    dimensions(1u, 1u, 1u);
    if (!AddMetalPipelineSourceRecipe(
            manifest, MetalStencilPipelineSourceRecipe(
                          step.operation.get<operation::Stencil>().plan))) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::Transform:
  case rund::kernel::NodeKind::Matrix:
  case rund::kernel::NodeKind::Factor:
  case rund::kernel::NodeKind::Solve:
  case rund::kernel::NodeKind::Spectrum:
    dimensions(1u, 1u, 1u);
    if (!AddMetalPipelineSourceRecipe(manifest,
                                      MetalNumericPipelineSourceRecipe())) {
      return manifest;
    }
    break;
  case rund::kernel::NodeKind::ScatterReduce:
    dimensions(1u, 3u, 1u);
    if (!AddMetalPipelineSourceRecipe(
            manifest,
            MetalScatterReducePipelineSourceRecipe(
                step.operation.get<operation::ScatterReduce>().plan))) {
      return manifest;
    }
    break;
  }
  const MetalStepControlShape control = PlanMetalStepControlShape(step, bound);
  if (!control.ok) {
    return manifest;
  }
  manifest.status_source_count = control.status_source_count;
  manifest.status_entry_count = control.status_entry_count;
  manifest.status_command_count = control.status_command_count;
  manifest.status_parameter_bytes = control.status_parameter_bytes;
  manifest.telemetry_source_count = control.telemetry_source_count;
  (void)CompleteMetalBackendManifest(manifest);
  return manifest;
}
#else
PreparedBackendManifest
BuildMetalBackendManifest(const KernelExecutionStep &,
                          const rund::kernel::ComputePlan &, const BoundStep *,
                          const std::uint64_t) noexcept {
  return {};
}
#endif

rund::AccelCheck PlanMetalPipelineStructureForCalibration(
    const MetalIcbCalibration &calibration,
    PreparedKernelPipelineReservation &reservation) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using backend_template_plan::add;
  using backend_template_plan::product;
  std::uint64_t commands = reservation.backend_dispatch_count;
  const std::uint64_t status_reset_command_count =
      reservation.backend_status_source_count == 0u ? 0u : 1u;
  for (const std::uint64_t count :
       {reservation.backend_reset_dispatch_count,
        reservation.backend_status_command_count,
        reservation.backend_telemetry_command_count, reservation.window_count,
        reservation.backend_publication_command_count,
        status_reset_command_count, std::uint64_t{2u}}) {
    if (!add(commands, count)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  std::uint64_t bindings = 0u;
  std::uint64_t binding_slot_upper =
      std::max(reservation.backend_command_binding_slot_upper,
               std::uint64_t{4u}); // pipeline open/close: indices 0...3
  if (reservation.backend_reset_dispatch_count != 0u) {
    binding_slot_upper = std::max(binding_slot_upper, std::uint64_t{2u});
  }
  if (status_reset_command_count != 0u) {
    binding_slot_upper = std::max(binding_slot_upper, std::uint64_t{3u});
  }
  if (reservation.backend_status_command_count != 0u) {
    // Status fold binds 0...5, plus step-control at 6 under profiling.
    binding_slot_upper = std::max(binding_slot_upper,
                                  reservation.backend_profile_step_count == 0u
                                      ? std::uint64_t{6u}
                                      : std::uint64_t{7u});
  }
  if (reservation.backend_telemetry_command_count != 0u) {
    binding_slot_upper = std::max(binding_slot_upper, std::uint64_t{9u});
  }
  if (reservation.window_count != 0u ||
      reservation.backend_publication_command_count != 0u ||
      reservation.nested_group_count != 0u) {
    // Window canonicalization/publication and the nested aggregate author
    // indices 0...7. Advance alone is a strict subset.
    binding_slot_upper = std::max(binding_slot_upper, std::uint64_t{8u});
  }
  std::uint64_t captured_binding_slots = binding_slot_upper;
  if (binding_slot_upper > kMetalPipelineGuardBinding ||
      !add(captured_binding_slots, 1u)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t parameter_bytes = reservation.backend_parameter_bytes;
  const MetalIcbChunkPlan icb_plan = PlanMetalIcbChunks(commands, calibration);
  if (!icb_plan.ok || icb_plan.chunk_count == 0u ||
      icb_plan.allocated_bytes == 0u) {
    return rund::AccelCheck{false, "accel_metal_icb_calibration_failed"};
  }
  std::uint64_t host = sizeof(MetalSequence);
  std::uint64_t bytes = 0u;
  MetalCaptureRowCapacity command_rows{};
  MetalCaptureRowCapacity binding_rows{};
  if (!add(host, icb_plan.retained_chunk_bytes) ||
      !product(commands, captured_binding_slots, bindings) ||
      !(command_rows = PlanMetalCaptureRowCapacity(commands)).ok ||
      !(binding_rows = PlanMetalCaptureRowCapacity(bindings)).ok ||
      !product(command_rows.rows, sizeof(MetalCommand), bytes) ||
      !add(host, bytes) ||
      !product(binding_rows.rows, sizeof(MetalCommandBinding), bytes) ||
      !add(host, bytes) ||
      !product(commands, sizeof(id<MTLComputePipelineState>), bytes) ||
      !add(host, bytes) || !product(bindings, sizeof(id<MTLResource>), bytes) ||
      !add(host, bytes) ||
      !product(reservation.backend_step_description_count,
               sizeof(MetalPipelineTelemetryRecord), bytes) ||
      !add(host, bytes) ||
      !product(reservation.backend_profile_step_count,
               sizeof(PreparedPipelineStepEvidence), bytes) ||
      !add(host, bytes) ||
      !product(reservation.nested_group_count, sizeof(std::shared_ptr<void>),
               bytes) ||
      !add(host, bytes) || !add(reservation.host_bytes, host)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  // Fixed inline payload is bounded per command, not per argument slot. Most
  // encoders bind one params record. The only multi-setBytes commands are Scan
  // (at most six aligned scalars), telemetry (params + two scalars), status
  // reset, and status fold. Their totals fit this row; variable status/reset
  // arrays are occurrence-weighted separately above. Multiplying by all 31
  // argument slots manufactured nearly 2 GiB of unreachable parameter memory
  // in the large recurrence product case.
  constexpr std::uint64_t FixedParameterUpper =
      sizeof(MetalNestedAggregateParams);
  static_assert(FixedParameterUpper == 176u);
  static_assert(FixedParameterUpper >= sizeof(MetalPublishParams));
  static_assert(FixedParameterUpper >= sizeof(MetalWindowParams));
  static_assert(FixedParameterUpper >= sizeof(MetalPipelineTelemetryParams));
  static_assert(FixedParameterUpper >= sizeof(MetalPipelineStatusParams));
  static_assert(FixedParameterUpper >= sizeof(NumericParams));
  static_assert(FixedParameterUpper >= sizeof(reset::Params));
  static_assert(FixedParameterUpper >= 6u * 16u);
  static_assert(FixedParameterUpper >= 80u + 16u + 16u);
  std::uint64_t fixed_parameter_bytes = 0u;
  if (!product(commands, FixedParameterUpper, fixed_parameter_bytes) ||
      !add(parameter_bytes, fixed_parameter_bytes)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (status_reset_command_count != 0u) {
    std::uint64_t reset_rows = 0u;
    std::uint64_t aligned = 0u;
    if (!product(reservation.backend_status_source_count,
                 sizeof(MetalPipelineResetMeta), reset_rows) ||
        !AddAlignedMetalParameterBytes(aligned, reset_rows) ||
        !add(parameter_bytes, aligned)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  if (!add(reservation.native_bytes, parameter_bytes) ||
      !add(reservation.native_bytes, icb_plan.allocated_bytes)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  const MetalPointerIdentityIndexLayout identity_index =
      PlanMetalPointerIdentityIndex(std::max(commands, bindings));
  if (!identity_index.ok) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::uint64_t native_window_bytes = 0u;
  std::uint64_t finalizer_transient_bytes = identity_index.byte_count;
  std::uint64_t status_binding_capacity = 0u;
  std::uint64_t status_description_bytes = 0u;
  if (!product(reservation.window_count, sizeof(MetalWindow),
               native_window_bytes) ||
      !add(finalizer_transient_bytes, native_window_bytes) ||
      !product(reservation.backend_step_description_count,
               kMetalPipelineStatusBindingCapacity, status_binding_capacity) ||
      !product(status_binding_capacity,
               sizeof(MetalPipelineStatusBindingRecord) +
                   sizeof(MetalPipelineStatusSourceMeta) +
                   sizeof(MetalPipelineResetMeta),
               status_description_bytes) ||
      !add(finalizer_transient_bytes, status_description_bytes) ||
      !product(reservation.backend_status_entry_count,
               sizeof(MetalPipelineStatusEntryMeta),
               status_description_bytes) ||
      !add(finalizer_transient_bytes, status_description_bytes) ||
      !product(reservation.backend_step_description_count,
               sizeof(PreparedProgramStatusSlice), status_description_bytes) ||
      !add(finalizer_transient_bytes, status_description_bytes) ||
      !add(finalizer_transient_bytes, parameter_bytes)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  // Resolved native window rows remain live until ICB finalization and
  // therefore coexist with the pointer-identity workspace, status-description
  // rows, and captured parameter bytes. Occurrence counts are conservative
  // frozen uppers for template-local description rows; transducer fusion may
  // use less, but no unplanned heap is hidden below this boundary.
  reservation.host_transient_bytes =
      std::max(reservation.host_transient_bytes, finalizer_transient_bytes);
  reservation.backend_command_count = commands;
  reservation.backend_command_chunk_count = icb_plan.chunk_count;
  reservation.backend_command_native_bytes = icb_plan.allocated_bytes;
  reservation.backend_command_binding_slot_upper = binding_slot_upper;
  reservation.backend_command_binding_count = bindings;
  reservation.backend_parameter_bytes = parameter_bytes;
  reservation.backend_window_control_command_count = reservation.window_count;
  // Retained command stream: the exact size-class ICB chunk count plus the
  // common control/status buffers; optional parameter, resident-state, and
  // profile buffers are explicit.
  reservation.backend_native_buffer_count =
      3u + static_cast<std::uint64_t>(parameter_bytes != 0u) +
      static_cast<std::uint64_t>(reservation.window_count != 0u) +
      static_cast<std::uint64_t>(reservation.backend_profile_step_count != 0u);
  reservation.backend_native_object_count =
      icb_plan.chunk_count + reservation.backend_native_buffer_count;
  if (!add(reservation.native_allocation_count,
           reservation.backend_native_object_count)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)calibration;
  (void)reservation;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck PlanMetalPipelineStructure(
    const rund::AccelContext &context,
    PreparedKernelPipelineReservation &reservation) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const ContextAdmission admission = AdmitContextForSupport(context);
  MetalAdapter *const adapter = admission.pick == nullptr
                                    ? nullptr
                                    : MetalAdapterFromPick(admission.pick->raw);
  if (!admission.check.ok || admission.api != rund::AccelApi::Metal ||
      adapter == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  if (!ValidMetalIcbCalibration(adapter->pipeline_icb_calibration)) {
    return rund::AccelCheck{false, "accel_metal_icb_calibration_failed"};
  }
  return PlanMetalPipelineStructureForCalibration(
      adapter->pipeline_icb_calibration, reservation);
#else
  (void)context;
  (void)reservation;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck RunMetalKernel(const BackendRun &run) {
  if (run.pick == nullptr || run.steps == nullptr || run.step_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::shared_ptr<void> prepared{};
  PreparedMemory memory{};
  const rund::AccelCheck ready = PrepareMetalResources(
      *run.pick, run.steps, run.step_count, run.final_dispatch_count,
      KernelPreparationMode::Standalone, run.resets, run.views, run.view_binds,
      nullptr, nullptr, nullptr, run.failed_node, prepared, memory);
  if (ready.ok && run.traffic != nullptr) {
    *run.traffic = MetalKernelTraffic(prepared);
  }
  return ready.ok ? RunMetalResources(*run.pick, prepared) : ready;
}

rund::AccelCheck PrepareMetalKernel(const BackendRun &run,
                                    std::shared_ptr<void> &prepared,
                                    PreparedMemory &memory) {
  prepared.reset();
  memory = {};
  if (run.pick == nullptr || run.steps == nullptr || run.step_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  return PrepareMetalResources(
      *run.pick, run.steps, run.step_count, run.final_dispatch_count,
      KernelPreparationMode::Standalone, run.resets, run.views, run.view_binds,
      nullptr, nullptr, nullptr, run.failed_node, prepared, memory);
}

rund::AccelCheck
PrepareMetalPipelinePrivateKernel(const BackendRun &run,
                                  std::shared_ptr<void> &prepared,
                                  PreparedMemory &memory) {
  prepared.reset();
  memory = {};
  if (run.pick == nullptr || run.steps == nullptr || run.step_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  return PrepareMetalResources(
      *run.pick, run.steps, run.step_count, run.final_dispatch_count,
      KernelPreparationMode::PipelinePrivate, run.resets, run.views,
      run.view_binds, run.scratch, &run, run.templates, run.failed_node,
      prepared, memory);
}

rund::AccelCheck PlanMetalPipelinePrivateKernel(
    const BackendRun &run,
    PreparedKernelRouteReservation &reservation) noexcept {
  return backend_template_plan::plan(run, MetalBackendShape(), reservation);
}

rund::AccelCheck
PlanMetalPipelineProgram(const KernelExecution &execution,
                         const PreparedKernelProgramRoute &route,
                         PreparedKernelRouteReservation &reservation) noexcept {
  return backend_template_plan::plan_program(execution, route,
                                             MetalBackendShape(), reservation);
}

rund::AccelCheck PlanMetalPipelineRecurrence(
    const MapRecurrencePreparationPlan &plan,
    PreparedMapRecurrenceReservation &reservation) noexcept {
  reservation = {};
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using backend_template_plan::add;
  using backend_template_plan::product;
  if (!plan.ok) {
    return rund::AccelCheck{false, plan.reason == nullptr
                                       ? "compute_pipeline_recurrence_invalid"
                                       : plan.reason};
  }
  if (plan.group_count == 0u) {
    return plan.history_group_count == 0u
               ? rund::AccelCheck{true, "ok"}
               : rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const std::uint64_t terminal_group_count = plan.terminal_group_count();
  const bool terminal_variant = terminal_group_count != 0u;
  const bool history_variant = plan.history_group_count != 0u;
  const std::uint64_t template_count =
      static_cast<std::uint64_t>(terminal_variant) +
      static_cast<std::uint64_t>(history_variant);
  if (plan.authority == nullptr || plan.canonical_artifact == nullptr ||
      plan.canonical_artifact != &plan.authority->artifact ||
      plan.authority->kind() != rund::kernel::NodeKind::Map ||
      plan.history_group_count > plan.group_count || template_count == 0u ||
      template_count > 2u || !plan.plan.ok ||
      plan.plan.api != rund::kernel::ComputeApi::Metal ||
      plan.canonical_artifact->key.api != rund::kernel::ComputeApi::Metal ||
      plan.canonical_artifact->kind !=
          rund::kernel::LoweringArtifactKind::MetalSource ||
      (terminal_variant
           ? plan.terminal_template_group_capacity < terminal_group_count
           : plan.terminal_template_group_capacity != 0u) ||
      (history_variant
           ? plan.history_template_group_capacity < plan.history_group_count
           : plan.history_template_group_capacity != 0u) ||
      plan.window_count == 0u ||
      plan.window_count != plan.plan.dispatch_count ||
      plan.input_count != plan.plan.input_buffer_count ||
      plan.output_count != plan.plan.output_buffer_count ||
      plan.input_layouts().size() != plan.input_count ||
      plan.output_layouts().size() != plan.output_count ||
      plan.terminal_source.ok != terminal_variant ||
      plan.history_source.ok != history_variant) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  for (const PreparedKernelProgramBindingIdentity identity :
       plan.input_layouts()) {
    if (identity.element_bytes == 0u || identity.count == 0u ||
        identity.stride_bytes < identity.element_bytes ||
        identity.usage != rund::kernel::kResidentUsageRead) {
      return rund::AccelCheck{false, "compute_binding_mismatch"};
    }
  }
  for (const PreparedKernelProgramBindingIdentity identity :
       plan.output_layouts()) {
    if (identity.element_bytes == 0u || identity.count == 0u ||
        identity.stride_bytes < identity.element_bytes ||
        identity.usage != rund::kernel::kResidentUsageWrite) {
      return rund::AccelCheck{false, "compute_binding_mismatch"};
    }
  }

  std::uint64_t per_route_host = 0u;
  std::uint64_t window_bytes = 0u;
  std::uint64_t routes_host = 0u;
  std::uint64_t history_host = 0u;
  if (!AddMetalMapRouteHostBytes(per_route_host, plan.plan) ||
      !product(plan.window_count, sizeof(rund::kernel::ComputeDispatchWindow),
               window_bytes) ||
      !add(per_route_host, window_bytes) ||
      !product(per_route_host, plan.group_count, routes_host) ||
      !product(plan.history_group_count, sizeof(MapRecurrenceHistory),
               history_host) ||
      !add(routes_host, history_host) ||
      !product(plan.plan.param_bytes, plan.group_count,
               reservation.route_native_bytes)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.route_host_bytes = routes_host;

  const auto add_template = [&](const MapRecurrenceSourcePlan &source,
                                const bool history) noexcept {
    const MetalRecurrenceSourceReservation source_reservation =
        PlanMetalRecurrenceSource(source, plan.plan, history);
    std::uint64_t host = sizeof(MetalMapRecurrenceTemplate);
    if (!source_reservation.ok ||
        !AddMetalMapTemplateHostBytes(host, plan.plan, 0u) ||
        !add(host, source_reservation.source_storage_bytes) ||
        !add(reservation.template_host_bytes, host) ||
        !add(reservation.template_source_bytes,
             source_reservation.source_bytes) ||
        !add(reservation.template_native_allocation_count,
             source_reservation.native_object_count)) {
      return false;
    }
    reservation.source_transient_bytes = std::max(
        reservation.source_transient_bytes, source_reservation.transient_bytes);
    return true;
  };
  if ((terminal_variant && !add_template(plan.terminal_source, false)) ||
      (history_variant && !add_template(plan.history_source, true))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }

  reservation.group_count = plan.group_count;
  reservation.history_group_count = plan.history_group_count;
  reservation.template_count = template_count;
  reservation.terminal_template_group_capacity =
      plan.terminal_template_group_capacity;
  reservation.history_template_group_capacity =
      plan.history_template_group_capacity;
  reservation.route_step_count = plan.group_count;
  reservation.template_step_count = template_count;
  reservation.route_native_allocation_count = plan.group_count;
  if (reservation.group_count != plan.group_count ||
      reservation.history_group_count != plan.history_group_count ||
      reservation.template_count != template_count ||
      reservation.terminal_template_group_capacity !=
          plan.terminal_template_group_capacity ||
      reservation.history_template_group_capacity !=
          plan.history_template_group_capacity ||
      reservation.route_step_count != plan.group_count ||
      reservation.template_step_count != template_count) {
    reservation = {};
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)plan;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

bool SameMetalPipelineProgramTemplate(
    const KernelExecution &execution, const PreparedKernelProgramRoute &left,
    const PreparedKernelProgramRoute &right) noexcept {
  return backend_template_plan::same_program_template(execution, left, right,
                                                      1u);
}

bool SameMetalPipelineTemplate(const BackendRun &left,
                               const BackendRun &right) noexcept {
  return backend_template_plan::same_template(left, right, 1u);
}

rund::AccelCheck SubmitPreparedMetalKernel(
    const BackendRun &run, const std::shared_ptr<void> &prepared,
    const KernelCompletion completion, void *const user, PreparedMemoryMeter *,
    const std::shared_ptr<void> &) noexcept {
  return run.pick != nullptr && prepared != nullptr && completion != nullptr
             ? SubmitMetalResources(*run.pick, prepared, completion, user)
             : rund::AccelCheck{false, "accel_kernel_run_invalid"};
}

} // namespace rund::node::accel::detail
