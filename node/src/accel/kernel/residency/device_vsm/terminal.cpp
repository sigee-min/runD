#include "terminal.hpp"

#include "validation.hpp"

#include <kernel/core/checked.hpp>

#include <limits>

namespace rund::node::accel::detail {

DeviceVsmFinal
ClassifyDeviceVsmTerminal(const DeviceVsmRequest &request,
                          const DeviceVsmNativeExecution &native) noexcept {
  if (request.proof == nullptr) {
    return {};
  }
  const DeviceVsmProof &proof = *request.proof;
  const auto &counters = native.counters;
  const std::uint64_t pages = proof.geometry.page_count;
  const std::uint64_t submission_count =
      request.submission_control == nullptr
          ? 0u
          : request.submission_control->count();
  if (submission_count != 1u) {
    return {};
  }
  const std::uint64_t output_pages = device_vsm_output_page_count(proof);
  std::uint32_t expected_wavefront_steps = 0u;
  std::uint32_t expected_wavefront_trace = 0u;
  const bool graph = proof.topology == DeviceVsmTopology::GraphMapReduce ||
                     proof.topology == DeviceVsmTopology::GraphPointwise ||
                     proof.topology == DeviceVsmTopology::GraphResident;
  const bool resident = proof.topology == DeviceVsmTopology::GraphResident;
  const bool window = proof.topology == DeviceVsmTopology::Window;
  DeviceVsmWindowRingPlan window_ring_plan{};
  DeviceVsmWindowRingResult window_ring_result{};
  std::uint64_t expected_window_internal_dispatches = 0u;
  const bool window_ring = window && proof.window.ring.gpu_owned;
  const bool window_ring_valid =
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
  const bool graph_wavefront_exact =
      !graph ? true
             : device_vsm_graph_wavefront_expected(graph_wavefront,
                                                   expected_wavefront_steps,
                                                   expected_wavefront_trace) &&
                   counters[DeviceVsmGraphWavefrontStepsWord] ==
                       expected_wavefront_steps &&
                   counters[DeviceVsmGraphWavefrontTraceWord] ==
                       expected_wavefront_trace;
  const bool graph_tile_exact =
      resident
          ? native.encoded_tile_count == DeviceVsmGraphPhysicalDispatchCount
          : native.encoded_tile_count == 0u;
  const bool window_footprint_exact =
      window_ring
          ? counters[DeviceVsmWindowBoundaryTransitionsWord] == 0u &&
                counters[DeviceVsmWindowFootprintChecksumWord] == 0u
          : !window || (counters[DeviceVsmWindowBoundaryTransitionsWord] ==
                            proof.window.footprint.boundary_transition_count &&
                        counters[DeviceVsmWindowFootprintChecksumWord] ==
                            proof.window.footprint.projection_checksum);
  const bool window_ring_exact =
      !window_ring ||
      (window_ring_valid &&
       native.window_seed_dispatches == window_ring_result.page_count &&
       native.window_compute_dispatches == window_ring_result.page_count &&
       native.window_internal_dispatches ==
           expected_window_internal_dispatches &&
       counters[DeviceVsmRingReuseTransitionsWord] ==
           window_ring_result.page_count &&
       counters[DeviceVsmRingScheduleChecksumWord] ==
           window_ring_result.schedule_checksum &&
       counters[DeviceVsmRingRoundTripsWord] == window_ring_result.round_trips);
  const bool unused_topology_words_exact =
      graph || window ||
      (counters[DeviceVsmGraphWavefrontStepsWord] == 0u &&
       counters[DeviceVsmGraphWavefrontTraceWord] == 0u);
  const bool ring_schedule_exact =
      ring ? device_vsm_ring_schedule_expected(proof.geometry, proof.width,
                                               expected_ring_reuse,
                                               expected_ring_checksum) &&
                 device_vsm_ring_storage_expected(
                     proof.geometry, proof.width, proof.residents.count,
                     expected_ring_state_bytes, expected_ring_scratch_bytes) &&
                 device_vsm_ring_round_trips_expected(
                     proof.geometry, proof.residents.count,
                     expected_ring_round_trips) &&
                 counters[DeviceVsmRingReuseTransitionsWord] ==
                     expected_ring_reuse &&
                 counters[DeviceVsmRingScheduleChecksumWord] ==
                     expected_ring_checksum &&
                 counters[DeviceVsmRingRoundTripsWord] ==
                     expected_ring_round_trips
      : window_ring ? window_ring_valid &&
                          counters[DeviceVsmRingReuseTransitionsWord] ==
                              window_ring_result.page_count &&
                          counters[DeviceVsmRingScheduleChecksumWord] ==
                              window_ring_result.schedule_checksum &&
                          counters[DeviceVsmRingRoundTripsWord] ==
                              window_ring_result.round_trips
                    : counters[DeviceVsmRingReuseTransitionsWord] == 0u &&
                          counters[DeviceVsmRingScheduleChecksumWord] == 0u &&
                          counters[DeviceVsmRingRoundTripsWord] == 0u;
  const bool input_complete = native.result_acquired && counters[0u] == pages &&
                              counters[1u] == pages && counters[2u] == pages;
  const bool complete = submission_count == 1u && input_complete &&
                        counters[3u] == pages &&
                        graph_wavefront_exact && graph_tile_exact &&
                        window_footprint_exact && window_ring_exact &&
                        unused_topology_words_exact && ring_schedule_exact;
  const bool overflow =
      complete && proof.topology == DeviceVsmTopology::GraphMapReduce &&
      proof.graph_map_reduce.semantic.op == rund::kernel::ReduceOp::Sum &&
      counters[4u] == 0u && counters[5u] == 0u &&
      counters[DeviceVsmOverflowWord] == 1u;
  const bool scan_overflow = submission_count == 1u && input_complete &&
                             proof.topology == DeviceVsmTopology::Scan &&
                             counters[4u] == 0u && counters[5u] == 0u &&
                             counters[DeviceVsmOverflowWord] == 1u &&
                             counters[DeviceVsmFailedPageWord] < pages &&
                             counters[3u] == counters[DeviceVsmFailedPageWord];
  const bool reduce_overflow =
      submission_count == 1u && input_complete &&
      proof.topology == DeviceVsmTopology::Reduce &&
      proof.reduce.semantic.op == rund::kernel::ReduceOp::Sum &&
      counters[4u] == 0u && counters[5u] == 0u &&
      counters[DeviceVsmOverflowWord] == 1u &&
      counters[DeviceVsmFailedPageWord] < pages &&
      counters[3u] == counters[DeviceVsmFailedPageWord];
  const bool exact = complete && counters[4u] == output_pages &&
                     counters[5u] == output_pages &&
                     counters[DeviceVsmOverflowWord] == 0u;
  const bool known = exact || overflow || scan_overflow || reduce_overflow;
  const bool page_overflow = scan_overflow || reduce_overflow;
  std::uint64_t overlap = 0u;
  std::uint64_t input_bytes = 0u;
  const bool overlap_valid =
      device_vsm_overlap_reuse_bytes(proof.geometry, overlap);
  const bool input_bytes_valid = ::rund::kernel::checked::mul(
      proof.geometry.logical_bytes, proof.residents.input_count, input_bytes);
  if (proof.topology == DeviceVsmTopology::GraphResident) {
    input_bytes = 0u;
  }
  return DeviceVsmFinal{
      .check =
          exact      ? rund::AccelCheck{true, "ok"}
          : overflow ? rund::AccelCheck{false, "compute_reduce_sum_overflow"}
          : scan_overflow ? rund::AccelCheck{false, "compute_scan_sum_overflow"}
          : reduce_overflow
              ? rund::AccelCheck{false, "compute_reduce_sum_overflow"}
              : rund::AccelCheck{false, "compute_device_lost"},
      .terminal =
          known ? DeviceVsmTerminal::Known : DeviceVsmTerminal::UnknownMayWrite,
      .evidence =
          DeviceVsmEvidence{
              .proof = proof.identity,
              .token = request.token,
              .generation = request.generation,
              .nonce = request.nonce,
              .page_count = pages,
              .failed_page =
                  page_overflow
                      ? static_cast<std::uint64_t>(
                            counters[DeviceVsmFailedPageWord])
                      : std::numeric_limits<std::uint64_t>::max(),
              .generated_epochs = counters[0u],
              .completed_epochs = counters[3u],
              .graph_wavefront_steps =
                  graph ? counters[DeviceVsmGraphWavefrontStepsWord] : 0u,
              .graph_wavefront_trace =
                  graph ? counters[DeviceVsmGraphWavefrontTraceWord] : 0u,
              .canonical_boundary_transitions =
                  window_ring ? proof.window.footprint.boundary_transition_count
                  : window    ? counters[DeviceVsmWindowBoundaryTransitionsWord]
                              : 0u,
              .canonical_footprint_checksum =
                  window_ring ? proof.window.footprint.projection_checksum
                  : window    ? counters[DeviceVsmWindowFootprintChecksumWord]
                              : 0u,
              .ring_reuse_transitions =
                  ring          ? counters[DeviceVsmRingReuseTransitionsWord]
                  : window_ring ? counters[DeviceVsmRingReuseTransitionsWord]
                                : 0u,
              .ring_schedule_checksum =
                  (ring || window_ring)
                      ? counters[DeviceVsmRingScheduleChecksumWord]
                      : 0u,
              .ring_round_trips = (ring || window_ring)
                                      ? counters[DeviceVsmRingRoundTripsWord]
                                      : 0u,
              .ring_state_bytes = ring          ? expected_ring_state_bytes
                                  : window_ring ? window_ring_plan.state_bytes
                                                : 0u,
              .ring_scratch_bytes = ring ? expected_ring_scratch_bytes
                                    : window_ring
                                        ? window_ring_plan.scratch_bytes
                                        : 0u,
              .window_seed_dispatches = native.window_seed_dispatches,
              .window_compute_dispatches = native.window_compute_dispatches,
              .window_internal_dispatches = native.window_internal_dispatches,
              .forecasted_pages = counters[1u],
              .promoted_pages = counters[2u],
              .drained_pages = counters[4u],
              .persisted_pages = counters[5u],
              .overlap_reused_bytes = exact && overlap_valid ? overlap : 0u,
              .gpu_backing_read_bytes =
                  known && input_bytes_valid ? input_bytes : 0u,
              .gpu_backing_write_bytes = exact ? proof.output_bytes : 0u,
              .native_submit_count = submission_count,
              .epoch_native_submit_count = 0u,
              .payload_dispatch_count = 1u,
              .tile_dispatch_count = resident ? native.encoded_tile_count : 0u,
              .host_service_turn_count = 0u,
              .host_epoch_callback_count = 0u,
              .final_callback_count = 1u,
              .completed_ns = native.completed_ns,
              .kernel_ns = exact ? native.kernel_ns : 0u,
              .kernel_samples = exact ? native.kernel_samples : 0u,
              .submit_wait_ns = native.submit_wait_ns,
              .max_live_frames = proof.width,
              .native_check_reason = native.native_check_reason,
              .native_check_code = native.native_check_code,
              .result_mapped = native.result_mapped,
              .native_check_ok = native.native_check_ok,
              .result_acquired = native.result_acquired,
              .may_write = !scan_overflow && !reduce_overflow,
          },
  };
}

} // namespace rund::node::accel::detail
