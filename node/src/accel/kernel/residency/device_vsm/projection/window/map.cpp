#include "local.hpp"

#include "../../source/typed_map/scalar.hpp"

namespace rund::node::accel::detail::device_vsm_window_projection::
    window_detail {
namespace {

[[nodiscard]] bool project_map(const BoundStep &step,
                               WindowMapAuthority &authority) noexcept {
  authority = {};
  if (!BoundStepMatches(step, rund::kernel::NodeKind::Map) ||
      step.control.active() || !step.resets.empty() || step.step == nullptr ||
      step.planned == nullptr || !step.step->map_semantic.recurrence_total ||
      step.step->artifact.metadata.read_count != 1u ||
      step.step->artifact.metadata.write_count != 1u ||
      step.planned->plan.input_buffer_count != 1u ||
      step.planned->plan.output_buffer_count != 1u || !step.map_windows.ok ||
      step.map_windows.size() == 0u) {
    return false;
  }
  const rund::kernel::BindingSet bindings = MapBindingFor(step);
  if (!bindings.ok || bindings.resident_inputs.count != 1u ||
      bindings.resident_outputs.count != 1u ||
      bindings.param_data_bytes != bindings.param_bytes ||
      bindings.param_bytes != step.planned->plan.param_bytes ||
      (bindings.param_bytes != 0u && bindings.param_data == nullptr)) {
    return false;
  }
  const MapSemantic &semantic = step.step->map_semantic;
  if (semantic.kind == MapSemanticKind::AddWrapU32Immediate) {
    authority = WindowMapAuthority{
        .step = &step,
        .bindings = bindings,
        .map = DeviceVsmWindowMap{
            .kind = DeviceVsmWindowMapKind::AddWrapU32Immediate,
            .immediate = semantic.immediate,
        }};
    return true;
  }
  if (semantic.kind == MapSemanticKind::MulWrapU32Immediate) {
    authority = WindowMapAuthority{
        .step = &step,
        .bindings = bindings,
        .map = DeviceVsmWindowMap{
            .kind = DeviceVsmWindowMapKind::MulWrapU32Immediate,
            .immediate = semantic.immediate,
        }};
    return true;
  }
  if (bindings.param_bytes != 0u ||
      !device_vsm_typed_map::validate_parameter_free_total_u32_scalar(
          step.step->artifact, step.step->cpu_input)) {
    return false;
  }
  authority = WindowMapAuthority{
      .step = &step,
      .bindings = bindings,
      .map = DeviceVsmWindowMap{
          .kind = DeviceVsmWindowMapKind::CanonicalTotalU32,
          .source_hi = step.step->artifact.key.op_hash_hi,
          .source_lo = step.step->artifact.key.op_hash_lo,
          .canonical_hi = step.step->artifact.key.canonical_ir_hash_hi,
          .canonical_lo = step.step->artifact.key.canonical_ir_hash_lo,
      }};
  return true;
}

} // namespace

bool project_prefix_map(const prepared::RunState &run, std::size_t &cursor,
                        WindowMapAuthority &authority, DeviceVsmWindowMap &map,
                        const char *&reason) {
  if (cursor >= run.bound.run.step_count ||
      !BoundStepMatches(run.bound.run.steps[cursor],
                        rund::kernel::NodeKind::Map)) {
    return true;
  }
  if (!project_map(run.bound.run.steps[cursor], authority)) {
    reason = "device_vsm_window_prefix_map_invalid";
    return false;
  }
  map = authority.map;
  ++cursor;
  return true;
}

bool project_suffix_map(const prepared::RunState &run, std::size_t &cursor,
                        WindowMapAuthority &authority, DeviceVsmWindowMap &map,
                        const char *&reason) {
  if (cursor >= run.bound.run.step_count) {
    return true;
  }
  if (!project_map(run.bound.run.steps[cursor], authority)) {
    reason = "device_vsm_window_suffix_map_invalid";
    return false;
  }
  map = authority.map;
  ++cursor;
  return true;
}

} // namespace
  // rund::node::accel::detail::device_vsm_window_projection::window_detail
