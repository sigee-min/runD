#include "../../../context/internal/support.hpp"
#include "../../../kernel/backend/execute.hpp"
#include "../../../kernel/backend/template/arithmetic.hpp"
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
#include "../../pipeline/source/recipe.hpp"
#include "../../range/local.hpp"
#include "../../reduce/local.hpp"
#include "../../runtime/map/source/upper.hpp"
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
#include "../pipeline/identity/index.hpp"
#include "parameter.hpp"
#include "source/recipe.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

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
    // Partition's embedded Scan block stage authors bindings 0...10.
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
  case rund::kernel::NodeKind::Window: {
    const RangePlan &range = *RangePlanFor(step.operation);
    out = RangeDescriptorCount(range);
    if (range.shape().resident_counted()) {
      out = std::max(out, std::uint64_t{4u});
    }
    break;
  }
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
  case rund::kernel::NodeKind::Window: {
    const RangePlan *const range = RangePlanFor(step.operation);
    const bool resident =
        range != nullptr && range->ok() && range->shape().resident_counted();
    valid = !resident || (active_control && status(1u, 1u, 0u));
    result.telemetry_source_count = resident ? 1u : 0u;
    break;
  }
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
  if (!PlanMetalCaptureBindingSlotUpper(step, plan, controlled, has_checks,
                                        manifest.capture_binding_slot_upper) ||
      !AddMetalStepSourceRecipes(step, plan, controlled, has_checks,
                                 manifest)) {
    return manifest;
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

} // namespace rund::node::accel::detail
