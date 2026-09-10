#include "request.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>
#include <string_view>

namespace rund::node::accel::detail {

[[nodiscard]] bool
device_vsm_capable(const DeviceVsmCapability &capability) noexcept {
  return capability.check.ok && capability.width >= 2u &&
         capability.width <= 4u && capability.gpu_addressable_backing &&
         capability.device_generated_recurrence &&
         capability.fixed_native_storage && capability.fixed_common_storage &&
         capability.one_native_submit && capability.host_service_turns_zero &&
         capability.host_epoch_callbacks_zero &&
         capability.aggregate_terminal_once && capability.bounded_page_io &&
         capability.physical_ring_storage;
}

[[nodiscard]] bool
device_vsm_request_valid(const DeviceVsmCapability &capability,
                         const DeviceVsmRequest &request) noexcept {
  return device_vsm_capable(capability) && request.proof != nullptr &&
         device_vsm_proof_valid(*request.proof) &&
         capability.width == request.proof->width &&
         capability.fixed_common_storage ==
             request.proof->fixed_common_storage &&
         request.lowering != nullptr && request.admission != nullptr &&
         request.token != 0u && request.generation != 0u &&
         request.nonce != 0u && request.submission_control != nullptr &&
         request.submission_control->count() == 0u &&
         request.final != nullptr &&
         request.user != nullptr;
}

[[nodiscard]] bool
device_vsm_final_valid(const DeviceVsmRequest &request,
                       const DeviceVsmFinal &final) noexcept {
  if (request.proof == nullptr) {
    return false;
  }
  const DeviceVsmProof &proof = *request.proof;
  const std::uint64_t submission_count =
      request.submission_control == nullptr
          ? 0u
          : request.submission_control->count();
  const DeviceVsmEvidence &evidence = final.evidence;
  std::uint64_t input_bytes = 0u;
  std::uint64_t overlap = 0u;
  std::uint32_t expected_wavefront_steps = 0u;
  std::uint32_t expected_wavefront_trace = 0u;
  const bool graph = proof.topology == DeviceVsmTopology::GraphMapReduce ||
                     proof.topology == DeviceVsmTopology::GraphPointwise ||
                     proof.topology == DeviceVsmTopology::GraphResident;
  const bool resident = proof.topology == DeviceVsmTopology::GraphResident;
  const bool window = proof.topology == DeviceVsmTopology::Window;
  const bool window_ring = window && proof.window.ring.gpu_owned;
  DeviceVsmWindowRingPlan window_ring_plan{};
  DeviceVsmWindowRingResult window_ring_result{};
  std::uint64_t window_internal_dispatch_limit = 0u;
  std::uint64_t expected_window_internal_dispatches = 0u;
  const bool window_internal_dispatch_limit_valid =
      device_vsm_window_internal_dispatch_count(
          proof.geometry.page_count, window_internal_dispatch_limit);
  const bool expected_window_ring =
      !window_ring ||
      (device_vsm_window_ring_plan_expected(proof.geometry, window_ring_plan) &&
       proof.window.ring == window_ring_plan &&
       device_vsm_window_ring_result_expected(proof.geometry, window_ring_plan,
                                              window_ring_result) &&
       device_vsm_window_internal_dispatch_count(
           window_ring_result.page_count,
           expected_window_internal_dispatches));
  const bool ring = proof.topology == DeviceVsmTopology::Pointwise ||
                    proof.topology == DeviceVsmTopology::GraphPointwise;
  std::uint32_t expected_ring_reuse = 0u;
  std::uint32_t expected_ring_checksum = 0u;
  std::uint32_t expected_ring_round_trips = 0u;
  std::uint64_t expected_ring_state_bytes = 0u;
  std::uint64_t expected_ring_scratch_bytes = 0u;
  const DeviceVsmGraphWavefrontProof &graph_wavefront =
      proof.topology == DeviceVsmTopology::GraphPointwise
          ? proof.graph_pointwise.wavefront
      : proof.topology == DeviceVsmTopology::GraphResident
          ? proof.graph_wavefront
          : proof.graph_map_reduce.wavefront;
  const bool expected_wavefront =
      graph && device_vsm_graph_wavefront_expected(graph_wavefront,
                                                   expected_wavefront_steps,
                                                   expected_wavefront_trace);
  const bool tile_possible = resident ? evidence.tile_dispatch_count <=
                                            DeviceVsmGraphPhysicalDispatchCount
                                      : evidence.tile_dispatch_count == 0u;
  const bool wavefront_possible =
      graph
          ? expected_wavefront &&
                ((evidence.graph_wavefront_steps == 0u &&
                  evidence.graph_wavefront_trace == 0u) ||
                 (evidence.graph_wavefront_steps == expected_wavefront_steps &&
                  evidence.graph_wavefront_trace == expected_wavefront_trace))
          : evidence.graph_wavefront_steps == 0u &&
                evidence.graph_wavefront_trace == 0u;
  const bool footprint_possible =
      window ? ((evidence.canonical_boundary_transitions == 0u &&
                 evidence.canonical_footprint_checksum == 0u) ||
                (evidence.canonical_boundary_transitions ==
                     proof.window.footprint.boundary_transition_count &&
                 evidence.canonical_footprint_checksum ==
                     proof.window.footprint.projection_checksum))
             : evidence.canonical_boundary_transitions == 0u &&
                   evidence.canonical_footprint_checksum == 0u;
  const bool expected_ring =
      ring &&
      device_vsm_ring_schedule_expected(proof.geometry, proof.width,
                                        expected_ring_reuse,
                                        expected_ring_checksum) &&
      device_vsm_ring_storage_expected(
          proof.geometry, proof.width, proof.residents.count,
          expected_ring_state_bytes, expected_ring_scratch_bytes) &&
      device_vsm_ring_round_trips_expected(
          proof.geometry, proof.residents.count, expected_ring_round_trips);
  const bool ring_possible =
      ring ? expected_ring &&
                 evidence.ring_reuse_transitions <= expected_ring_reuse &&
                 evidence.ring_round_trips <= expected_ring_round_trips &&
                 evidence.ring_state_bytes == expected_ring_state_bytes &&
                 evidence.ring_scratch_bytes == expected_ring_scratch_bytes
      : window_ring
          ? expected_window_ring &&
                (final.terminal == DeviceVsmTerminal::Known
                     ? evidence.ring_reuse_transitions ==
                           window_ring_result.page_count
                     : evidence.ring_reuse_transitions <=
                           window_ring_result.page_count) &&
                (final.terminal == DeviceVsmTerminal::Known
                     ? evidence.ring_schedule_checksum ==
                           window_ring_result.schedule_checksum
                     : evidence.ring_schedule_checksum == 0u ||
                           evidence.ring_schedule_checksum ==
                               window_ring_result.schedule_checksum) &&
                evidence.ring_round_trips <= window_ring_result.round_trips &&
                evidence.ring_state_bytes == window_ring_plan.state_bytes &&
                evidence.ring_scratch_bytes == window_ring_plan.scratch_bytes
          : evidence.ring_reuse_transitions == 0u &&
                evidence.ring_schedule_checksum == 0u &&
                evidence.ring_round_trips == 0u &&
                evidence.ring_state_bytes == 0u &&
                evidence.ring_scratch_bytes == 0u;
  const bool scan_overflow =
      final.terminal == DeviceVsmTerminal::Known &&
      proof.topology == DeviceVsmTopology::Scan &&
      final.check.reason != nullptr &&
      std::string_view{final.check.reason} == "compute_scan_sum_overflow";
  const bool reduce_overflow =
      final.terminal == DeviceVsmTerminal::Known &&
      proof.topology == DeviceVsmTopology::Reduce &&
      proof.reduce.semantic.op == rund::kernel::ReduceOp::Sum &&
      final.check.reason != nullptr &&
      std::string_view{final.check.reason} == "compute_reduce_sum_overflow";
  const bool failed_page_valid =
      scan_overflow || reduce_overflow
          ? evidence.failed_page < evidence.page_count &&
                evidence.failed_page == evidence.completed_epochs
          : evidence.failed_page == std::numeric_limits<std::uint64_t>::max();
  if ((!resident && !::rund::kernel::checked::mul(proof.geometry.logical_bytes,
                                                 proof.residents.input_count,
                                                 input_bytes)) ||
      !device_vsm_overlap_reuse_bytes(proof.geometry, overlap) ||
      !wavefront_possible || !tile_possible || !footprint_possible ||
      !window_internal_dispatch_limit_valid ||
      !ring_possible || !expected_window_ring ||
      evidence.proof != proof.identity || evidence.token != request.token ||
      evidence.generation != request.generation ||
      evidence.nonce != request.nonce ||
      evidence.page_count != proof.geometry.page_count ||
      evidence.generated_epochs > evidence.page_count ||
      evidence.completed_epochs > evidence.generated_epochs ||
      evidence.forecasted_pages > evidence.page_count ||
      evidence.promoted_pages > evidence.forecasted_pages ||
      evidence.drained_pages > evidence.completed_epochs ||
      evidence.persisted_pages > evidence.drained_pages ||
      evidence.overlap_reused_bytes > overlap ||
      evidence.gpu_backing_read_bytes > input_bytes ||
      evidence.gpu_backing_write_bytes > proof.output_bytes ||
      request.submission_control == nullptr || submission_count != 1u ||
      evidence.native_submit_count != submission_count ||
      evidence.epoch_native_submit_count != 0u ||
      evidence.payload_dispatch_count != 1u ||
      evidence.host_service_turn_count != 0u ||
      evidence.host_epoch_callback_count != 0u ||
      evidence.final_callback_count != 1u || evidence.max_live_frames == 0u ||
      evidence.max_live_frames > proof.width ||
      evidence.window_seed_dispatches > proof.geometry.page_count ||
      evidence.window_compute_dispatches > proof.geometry.page_count ||
      evidence.window_internal_dispatches > window_internal_dispatch_limit ||
      !failed_page_valid) {
    return false;
  }
  if (final.check.ok) {
    const std::uint64_t output_pages = device_vsm_output_page_count(proof);
    return final.terminal == DeviceVsmTerminal::Known && evidence.may_write &&
           (!graph ||
            (evidence.graph_wavefront_steps == expected_wavefront_steps &&
             evidence.graph_wavefront_trace == expected_wavefront_trace)) &&
           (!resident || evidence.tile_dispatch_count ==
                             DeviceVsmGraphPhysicalDispatchCount) &&
           (!window || (evidence.canonical_boundary_transitions ==
                            proof.window.footprint.boundary_transition_count &&
                        evidence.canonical_footprint_checksum ==
                            proof.window.footprint.projection_checksum)) &&
           (!ring ||
            (evidence.ring_reuse_transitions == expected_ring_reuse &&
             evidence.ring_schedule_checksum == expected_ring_checksum &&
             evidence.ring_round_trips == expected_ring_round_trips)) &&
           (!window_ring ||
            (evidence.window_seed_dispatches == proof.geometry.page_count &&
             evidence.window_compute_dispatches == proof.geometry.page_count &&
             evidence.window_internal_dispatches ==
                 expected_window_internal_dispatches &&
             evidence.ring_reuse_transitions == window_ring_result.page_count &&
             evidence.ring_schedule_checksum ==
                 window_ring_result.schedule_checksum &&
             evidence.ring_round_trips == window_ring_result.round_trips)) &&
           evidence.generated_epochs == evidence.page_count &&
           evidence.completed_epochs == evidence.page_count &&
           evidence.forecasted_pages == evidence.page_count &&
           evidence.promoted_pages == evidence.page_count &&
           evidence.drained_pages == output_pages &&
           evidence.persisted_pages == output_pages &&
           evidence.overlap_reused_bytes == overlap &&
           evidence.gpu_backing_read_bytes == input_bytes &&
           evidence.gpu_backing_write_bytes == proof.output_bytes &&
           evidence.max_live_frames ==
               std::min<std::uint64_t>(proof.width, evidence.page_count);
  }
  if (final.terminal == DeviceVsmTerminal::UnknownMayWrite) {
    return evidence.may_write;
  }
  if (proof.topology == DeviceVsmTopology::GraphMapReduce &&
      proof.graph_map_reduce.semantic.op == rund::kernel::ReduceOp::Sum &&
      final.check.reason != nullptr &&
      std::string_view{final.check.reason} == "compute_reduce_sum_overflow") {
    return final.terminal == DeviceVsmTerminal::Known && evidence.may_write &&
           evidence.graph_wavefront_steps == expected_wavefront_steps &&
           evidence.graph_wavefront_trace == expected_wavefront_trace &&
           evidence.generated_epochs == evidence.page_count &&
           evidence.completed_epochs == evidence.page_count &&
           evidence.forecasted_pages == evidence.page_count &&
           evidence.promoted_pages == evidence.page_count &&
           evidence.drained_pages == 0u && evidence.persisted_pages == 0u &&
           evidence.overlap_reused_bytes == 0u &&
           evidence.gpu_backing_read_bytes == input_bytes &&
           evidence.gpu_backing_write_bytes == 0u &&
           evidence.max_live_frames ==
               std::min<std::uint64_t>(proof.width, evidence.page_count);
  }
  if (proof.topology == DeviceVsmTopology::Scan &&
      final.check.reason != nullptr &&
      std::string_view{final.check.reason} == "compute_scan_sum_overflow") {
    return final.terminal == DeviceVsmTerminal::Known && !evidence.may_write &&
           evidence.generated_epochs == evidence.page_count &&
           evidence.completed_epochs < evidence.page_count &&
           evidence.forecasted_pages == evidence.page_count &&
           evidence.promoted_pages == evidence.page_count &&
           evidence.drained_pages == 0u && evidence.persisted_pages == 0u &&
           evidence.overlap_reused_bytes == 0u &&
           evidence.gpu_backing_read_bytes == input_bytes &&
           evidence.gpu_backing_write_bytes == 0u &&
           evidence.max_live_frames ==
               std::min<std::uint64_t>(proof.width, evidence.page_count);
  }
  if (proof.topology == DeviceVsmTopology::Reduce &&
      final.check.reason != nullptr &&
      std::string_view{final.check.reason} == "compute_reduce_sum_overflow") {
    return final.terminal == DeviceVsmTerminal::Known && !evidence.may_write &&
           evidence.generated_epochs == evidence.page_count &&
           evidence.completed_epochs < evidence.page_count &&
           evidence.forecasted_pages == evidence.page_count &&
           evidence.promoted_pages == evidence.page_count &&
           evidence.drained_pages == 0u && evidence.persisted_pages == 0u &&
           evidence.overlap_reused_bytes == 0u &&
           evidence.gpu_backing_read_bytes == input_bytes &&
           evidence.gpu_backing_write_bytes == 0u &&
           evidence.max_live_frames ==
               std::min<std::uint64_t>(proof.width, evidence.page_count);
  }
  std::uint64_t completed_ring_round_trips = 0u;
  if (ring && !::rund::kernel::checked::mul(evidence.completed_epochs,
                                           proof.residents.count,
                                           completed_ring_round_trips)) {
    return false;
  }
  return final.terminal == DeviceVsmTerminal::Known &&
         (evidence.completed_epochs == 0u || evidence.may_write) &&
         (!ring || (evidence.ring_reuse_transitions == expected_ring_reuse &&
                    evidence.ring_schedule_checksum == expected_ring_checksum &&
                    evidence.ring_round_trips == completed_ring_round_trips));
}

} // namespace rund::node::accel::detail
