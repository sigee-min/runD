#pragma once

#include "capability.hpp"
#include "request.hpp"

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail {

[[nodiscard]] inline bool service_free_direct_capable(
    const ServiceFreeDirectCapability &capability) noexcept {
  return capability.check.ok && capability.device_generated_recurrence &&
         capability.fixed_native_storage && capability.fixed_common_storage &&
         capability.one_native_submit && capability.host_service_turns_zero &&
         capability.host_epoch_callbacks_zero &&
         capability.aggregate_terminal_once &&
         (capability.terminal_output || capability.history_output);
}

[[nodiscard]] inline bool
service_free_direct_proof_valid(const ServiceFreeDirectProof &proof) noexcept {
  if (!proof.identity || proof.semantic_owner == nullptr ||
      proof.artifact == nullptr || !proof.artifact->ok || !proof.plan.ok ||
      proof.plan.api == rund::kernel::ComputeApi::Cpu ||
      proof.artifact->key.api != proof.plan.api || proof.iterations < 2u ||
      proof.iterations > std::numeric_limits<std::uint32_t>::max() ||
      proof.input_count == 0u || proof.output_count == 0u ||
      !proof.fixed_common_storage || proof.state_count == 0u ||
      proof.state_count > proof.StateCapacity ||
      proof.input_count > proof.BindingCapacity ||
      proof.output_count > proof.BindingCapacity ||
      proof.plan.input_buffer_count != proof.input_count ||
      proof.plan.output_buffer_count != proof.output_count ||
      proof.windows == nullptr || proof.window_count == 0u ||
      proof.plan.dispatch_count != proof.window_count ||
      proof.parameter_bytes != proof.plan.param_bytes ||
      (proof.parameter_bytes != 0u && proof.parameters == nullptr)) {
    return false;
  }
  for (std::size_t index = 0u; index < proof.input_count; ++index) {
    if (proof.inputs[index].id == 0u || proof.inputs[index].bytes == 0u ||
        proof.inputs[index].usage != rund::kernel::kResidentUsageRead ||
        proof.input_handles[index] == nullptr) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < proof.output_count; ++index) {
    if (proof.outputs[index].id == 0u || proof.outputs[index].bytes == 0u ||
        proof.outputs[index].usage != rund::kernel::kResidentUsageWrite ||
        proof.output_handles[index] == nullptr) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < proof.state_count; ++index) {
    const rund::kernel::ResidentBufferRef &state = proof.states[index];
    if (state.id == 0u || state.bytes == 0u || state.offset_bytes != 0u ||
        state.element_bytes != 1u || state.stride_bytes != 1u ||
        state.count != state.bytes || state.usage == 0u ||
        (state.usage & ~(rund::kernel::kResidentUsageRead |
                         rund::kernel::kResidentUsageWrite)) != 0u ||
        proof.state_handles[index] == nullptr) {
      return false;
    }
    for (std::size_t previous = 0u; previous < index; ++previous) {
      if (proof.states[previous].id == state.id ||
          proof.state_handles[previous] == proof.state_handles[index]) {
        return false;
      }
    }
  }
  for (std::uint64_t index = 0u; index < proof.window_count; ++index) {
    if (proof.windows[index].tile_count == 0u) {
      return false;
    }
  }
  if (proof.retention == ServiceFreeDirectRetention::History) {
    for (std::size_t index = 0u; index < proof.output_count; ++index) {
      if (proof.output_pitch_bytes[index] == 0u) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] inline bool service_free_direct_request_valid(
    const ServiceFreeDirectCapability &capability,
    const ServiceFreeDirectRequest &request) noexcept {
  return service_free_direct_capable(capability) && request.proof != nullptr &&
         service_free_direct_proof_valid(*request.proof) &&
         capability.fixed_common_storage ==
             request.proof->fixed_common_storage &&
         request.lowering != nullptr && request.admission != nullptr &&
         request.token != 0u && request.generation != 0u &&
         request.nonce != 0u && request.final != nullptr &&
         request.user != nullptr &&
         ((request.proof->retention == ServiceFreeDirectRetention::Terminal &&
           capability.terminal_output) ||
          (request.proof->retention == ServiceFreeDirectRetention::History &&
           capability.history_output));
}

[[nodiscard]] inline bool
service_free_direct_final_valid(const ServiceFreeDirectRequest &request,
                                const ServiceFreeDirectFinal &final) noexcept {
  if (request.proof == nullptr ||
      final.evidence.proof != request.proof->identity ||
      final.evidence.token != request.token ||
      final.evidence.generation != request.generation ||
      final.evidence.nonce != request.nonce ||
      final.evidence.iterations != request.proof->iterations ||
      final.evidence.completed_iterations > final.evidence.iterations ||
      final.evidence.native_submit_count != 1u ||
      final.evidence.epoch_native_submit_count != 0u ||
      final.evidence.payload_dispatch_count != 1u ||
      final.evidence.host_service_turn_count != 0u ||
      final.evidence.host_epoch_callback_count != 0u ||
      final.evidence.final_callback_count != 1u) {
    return false;
  }
  if (final.check.ok) {
    return final.terminal == ServiceFreeDirectTerminal::Known &&
           final.evidence.completed_iterations == final.evidence.iterations &&
           final.evidence.may_write;
  }
  if (final.terminal == ServiceFreeDirectTerminal::UnknownMayWrite) {
    return final.evidence.may_write;
  }
  return final.terminal == ServiceFreeDirectTerminal::Known &&
         (final.evidence.completed_iterations == 0u ||
          final.evidence.may_write);
}

} // namespace rund::node::accel::detail
