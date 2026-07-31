#include "local.hpp"

#include "../../../plan/validation.hpp"

namespace rund::node::accel::detail {

rund::kernel::BindingSet BindMapStep(const KernelExecutionStep &step,
                                     const PlannedStep &planned,
                                     const StepBinds &binds) {
  const rund::kernel::ExecutionMetadata &metadata = MapMetadata(step);
  return rund::kernel::BindingSet{
      .phase_id = planned.plan.phase_id,
      .tile_count = planned.plan.tile_count,
      .lane_count = planned.plan.tile_count,
      .op_hash_hi = planned.plan.op_hash_hi,
      .op_hash_lo = planned.plan.op_hash_lo,
      .api = planned.plan.api,
      .scalar = planned.plan.scalar,
      .domain = planned.plan.domain,
      .input_bytes_per_tile = planned.plan.input_bytes_per_tile,
      .output_bytes_per_tile = planned.plan.output_bytes_per_tile,
      .param_bytes = planned.plan.param_bytes,
      .metadata_bytes_per_tile = planned.plan.metadata_bytes_per_tile,
      .input_element_bytes = metadata.input_element_bytes.empty()
                                 ? nullptr
                                 : metadata.input_element_bytes.data(),
      .input_element_byte_count =
          static_cast<rund::kernel::u64>(metadata.input_element_bytes.size()),
      .param_data = metadata.param_storage.empty()
                        ? nullptr
                        : metadata.param_storage.data(),
      .param_data_bytes =
          static_cast<rund::kernel::u64>(metadata.param_storage.size()),
      .sequence_tiles = nullptr,
      .sequence_tile_count = 0u,
      .ok = true,
      .reason = "ok",
      .resident_inputs = binds.inputs.range(),
      .resident_outputs = binds.outputs.range(),
  };
}

rund::kernel::BindingValidation
ValidateMapStepBindings(const KernelExecutionStep &step,
                        const PlannedStep &planned,
                        const rund::kernel::BindingSet &bindings) {
  const rund::kernel::ExecutionMetadata &metadata = MapMetadata(step);
  return BindingPlanCheck(planned.plan, bindings, metadata,
                          PlanBindingInputMode::MapResidentViews);
}

} // namespace rund::node::accel::detail
