#include "scan.hpp"
#include "window.hpp"

#include "../../../../scan/shape.hpp"
#include "../../../backend/run.hpp"
#include "../../../bindings/scan.hpp"
#include "../../../recurrence/match.hpp"

#include <kernel/program/compute/scan/plan.hpp>

namespace rund::node::accel::detail::device_vsm_scan_projection {
namespace {

[[nodiscard]] bool same_plan(const rund::kernel::ScanPlan &left,
                             const rund::kernel::ScanPlan &right) noexcept {
  return left.ok && right.ok && left.op == right.op &&
         left.element == right.element &&
         left.element_count == right.element_count &&
         left.element_bytes == right.element_bytes &&
         left.block_size == right.block_size &&
         left.block_count == right.block_count &&
         left.pass_count == right.pass_count &&
         left.temp_bytes == right.temp_bytes &&
         left.count_source == right.count_source;
}

[[nodiscard]] bool project_map(const BoundStep &step, DeviceVsmScanMap &map,
                               rund::kernel::BindingSet &bindings) noexcept {
  map = {};
  bindings = {};
  if (!BoundStepMatches(step, rund::kernel::NodeKind::Map) ||
      step.control.active() || !step.resets.empty() || step.step == nullptr ||
      step.planned == nullptr || !step.step->map_semantic.recurrence_total ||
      step.step->artifact.key.scalar != rund::kernel::ComputeScalar::Lane64 ||
      step.step->artifact.key.domain != rund::kernel::ComputeDomain::U64 ||
      step.step->artifact.metadata.read_count == 0u ||
      step.step->artifact.metadata.write_count != 1u ||
      !step.step->artifact.metadata.read_routes.empty() ||
      !step.step->cpu_input.ok ||
      step.step->cpu_input.key != step.step->artifact.key ||
      step.planned->plan.input_buffer_count !=
          step.step->artifact.metadata.read_count ||
      step.planned->plan.output_buffer_count != 1u || !step.map_windows.ok ||
      step.map_windows.size() == 0u) {
    return false;
  }
  bindings = MapBindingFor(step);
  if (!bindings.ok || bindings.param_data_bytes != bindings.param_bytes ||
      bindings.param_bytes != step.planned->plan.param_bytes ||
      (bindings.param_bytes != 0u && bindings.param_data == nullptr)) {
    return false;
  }
  if (step.step->map_semantic.kind == MapSemanticKind::AddWrapU64Immediate) {
    map = DeviceVsmScanMap{
        .kind = DeviceVsmScanMapKind::AddWrapU64Immediate,
        .immediate =
            static_cast<std::uint64_t>(step.step->map_semantic.immediate) |
            (static_cast<std::uint64_t>(step.step->map_semantic.maximum)
             << 32u),
    };
  } else {
    map = DeviceVsmScanMap{
        .kind = DeviceVsmScanMapKind::CanonicalTotalU64,
    };
  }
  return true;
}

[[nodiscard]] bool exact_scan(const BoundStep &step,
                              const rund::kernel::ComputeApi api,
                              Authority &authority) noexcept {
  authority = {};
  const auto *const active = OperationFor<operation::Scan>(step);
  const auto *const bindings =
      BindingsFor<ScanBinds>(step, rund::kernel::NodeKind::Scan);
  if (active == nullptr || bindings == nullptr || step.control.active() ||
      api == rund::kernel::ComputeApi::Cpu ||
      !ScanShapeOk(active->desc, active->plan) ||
      !ScanResidentShapeOk(active->plan, *bindings) ||
      (active->plan.element != rund::kernel::ScanElement::U32 &&
       active->plan.element != rund::kernel::ScanElement::U64) ||
      active->plan.count_source !=
          rund::kernel::ComputeCountSource::Descriptor) {
    return false;
  }
  authority = Authority{.semantic = active->plan, .api = api};
  return true;
}

[[nodiscard]] bool exact_step(const prepared::RunState *const run,
                              Authority &authority) noexcept {
  authority = {};
  if (run == nullptr || !run->bound.ok || run->bound.run.step_count == 0u ||
      run->bound.run.step_count > 2u || run->bound.run.steps == nullptr ||
      (run->bound.run.resets != nullptr && !run->bound.run.resets->empty())) {
    return false;
  }
  const rund::kernel::ComputeApi api =
      run->execution.context_admission.api == rund::AccelApi::Metal
          ? rund::kernel::ComputeApi::Metal
      : run->execution.context_admission.api == rund::AccelApi::Vulkan
          ? rund::kernel::ComputeApi::Vulkan
          : rund::kernel::ComputeApi::Cpu;
  std::size_t cursor = 0u;
  DeviceVsmScanMap map{};
  rund::kernel::BindingSet map_bindings{};
  const BoundStep *map_step = nullptr;
  if (run->bound.run.step_count == 2u) {
    map_step = &run->bound.run.steps[cursor++];
    if (!project_map(*map_step, map, map_bindings)) {
      return false;
    }
  }
  if (!exact_scan(run->bound.run.steps[cursor], api, authority)) {
    return false;
  }
  authority.map = map;
  authority.map_step = map_step;
  authority.map_bindings = map_bindings;
  return true;
}

} // namespace

bool exact(const prepared::PipelineState &pipeline, Authority &authority,
           const char *&reason) noexcept {
  authority = {};
  reason = "device_vsm_scan_pipeline_invalid";
  if (pipeline.state_count == 0u || pipeline.states == nullptr ||
      pipeline.size == 0u ||
      !exact_step(pipeline.states[0u].get(), authority)) {
    return false;
  }
  for (std::size_t index = 1u; index < pipeline.state_count; ++index) {
    Authority candidate{};
    if (!exact_step(pipeline.states[index].get(), candidate) ||
        candidate.api != authority.api || candidate.map != authority.map ||
        !same_plan(candidate.semantic, authority.semantic) ||
        !device_vsm_window_projection::same_parameters(
            candidate.map_bindings, authority.map_bindings) ||
        static_cast<bool>(candidate.map_step) !=
            static_cast<bool>(authority.map_step) ||
        (candidate.map_step != nullptr &&
         (!SamePlan(candidate.map_step->planned->plan,
                    authority.map_step->planned->plan) ||
          !SameArtifact(candidate.map_step->step->artifact,
                        authority.map_step->step->artifact) ||
          !SameWindows(*candidate.map_step, *authority.map_step)))) {
      reason = "device_vsm_scan_state_mismatch";
      return false;
    }
  }
  reason = "ok";
  return true;
}

bool same(const prepared::PipelineState &pipeline,
          const Authority &authority) noexcept {
  Authority candidate{};
  const char *reason = nullptr;
  return exact(pipeline, candidate, reason) && candidate.api == authority.api &&
         candidate.map == authority.map &&
         same_plan(candidate.semantic, authority.semantic) &&
         device_vsm_window_projection::same_parameters(
             candidate.map_bindings, authority.map_bindings) &&
         static_cast<bool>(candidate.map_step) ==
             static_cast<bool>(authority.map_step) &&
         (candidate.map_step == nullptr ||
          (SamePlan(candidate.map_step->planned->plan,
                    authority.map_step->planned->plan) &&
           SameArtifact(candidate.map_step->step->artifact,
                        authority.map_step->step->artifact) &&
           SameWindows(*candidate.map_step, *authority.map_step)));
}

} // namespace rund::node::accel::detail::device_vsm_scan_projection
