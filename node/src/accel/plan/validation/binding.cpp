#include "local.hpp"

#include <kernel/program/compute/binding/validation.hpp>
#include <kernel/program/compute/limit.hpp>
#include <kernel/program/compute/plan.hpp>

#include <array>

namespace rund::node::accel::detail {

rund::kernel::BindingValidation
BindingPlanCheck(const rund::kernel::ComputePlan &plan,
                 const rund::kernel::BindingSet &bindings,
                 const rund::kernel::ExecutionMetadata &metadata,
                 const PlanBindingInputMode input_mode) noexcept {
  if (!rund::kernel::ComputeScalarValid(plan.scalar)) {
    return rund::kernel::BindingValidation{.reason =
                                               "compute_binding_mismatch"};
  }
  if (metadata.input_element_bytes.size() >
      rund::kernel::kMaxComputeBindingCount) {
    return rund::kernel::BindingValidation{.reason =
                                               "compute_binding_mismatch"};
  }
  std::array<rund::kernel::u64, rund::kernel::kMaxComputeBindingCount>
      input_counts{};
  for (std::size_t index = 0u;
       index < metadata.input_element_bytes.size(); ++index) {
    input_counts[index] = rund::kernel::RequiredInputCount(
        metadata, static_cast<rund::kernel::u64>(index), plan.tile_count);
    if (input_counts[index] == 0u) {
      return rund::kernel::BindingValidation{
          .reason = "compute_binding_input_count_mismatch"};
    }
  }
  const rund::kernel::BindingValidation validation =
      rund::kernel::ValidateRuntimeBindings(
          bindings,
          rund::kernel::BindingObligations{
              .tile_count = plan.tile_count,
              .input_buffer_count = plan.input_buffer_count,
              .output_buffer_count = plan.output_buffer_count,
              .input_bytes_per_tile = plan.input_bytes_per_tile,
              .output_bytes_per_tile = plan.output_bytes_per_tile,
              .param_bytes = plan.param_bytes,
              .input_element_bytes = metadata.input_element_bytes.data(),
              .input_element_byte_count = static_cast<rund::kernel::u64>(
                  metadata.input_element_bytes.size()),
              .input_counts = input_counts.data(),
              .input_count_count = static_cast<rund::kernel::u64>(
                  metadata.input_element_bytes.size()),
              .output_element_bytes = metadata.output_element_bytes.data(),
              .output_element_byte_count = static_cast<rund::kernel::u64>(
                  metadata.output_element_bytes.size()),
              .allow_resident_stride =
                  input_mode == PlanBindingInputMode::MapResidentViews,
          });
  if (!validation.ok) {
    if (validation.reason == nullptr || ReasonIsOk(validation.reason)) {
      return rund::kernel::BindingValidation{.reason =
                                                 "compute_binding_mismatch"};
    }
    return validation;
  }
  if (bindings.phase_id == plan.phase_id &&
      bindings.tile_count == plan.tile_count &&
      bindings.op_hash_hi == plan.op_hash_hi &&
      bindings.op_hash_lo == plan.op_hash_lo && bindings.api == plan.api &&
      bindings.scalar == plan.scalar &&
      BindingInputCountMatches(plan, bindings, input_mode) &&
      bindings.input_bytes_per_tile == plan.input_bytes_per_tile &&
      bindings.output_bytes_per_tile == plan.output_bytes_per_tile &&
      bindings.param_bytes == plan.param_bytes &&
      bindings.metadata_bytes_per_tile == plan.metadata_bytes_per_tile) {
    return rund::kernel::BindingValidation{.ok = true, .reason = "ok"};
  }
  return rund::kernel::BindingValidation{.reason = "compute_binding_mismatch"};
}

} // namespace rund::node::accel::detail
