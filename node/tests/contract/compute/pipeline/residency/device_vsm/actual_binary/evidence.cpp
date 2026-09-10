#include "internal.hpp"

namespace rund_node_test_pipeline_residency::device_vsm_test::binary_detail {

[[nodiscard]] bool ValidateBinaryOutputAndEvidence(PreparedBinary &test_case,
                                                   std::uint64_t &retained) {
  wait_detail::release_self(*test_case.wait);
  std::vector<std::uint32_t> observed(test_case.elements);
  const accel::AccelTransfer downloaded = accel::DownloadAccelBufferMeasured(
      test_case.native->context, test_case.buffers[2u], observed.data(),
      observed.size() * sizeof(std::uint32_t));
  if (!downloaded.check.ok) {
    return false;
  }
  for (std::size_t index = 0u; index < observed.size(); ++index) {
    if (observed[index] != test_case.first[index] + test_case.second[index]) {
      return false;
    }
  }
  retained = test_case.preparation.capability.retained_bytes;
  const accel::DeviceVsmEvidence &evidence = test_case.wait->final.evidence;
  std::uint32_t expected_ring_reuse = 0u;
  std::uint32_t expected_ring_checksum = 0u;
  std::uint32_t expected_ring_round_trips = 0u;
  std::uint64_t expected_ring_state_bytes = 0u;
  std::uint64_t expected_ring_scratch_bytes = 0u;
  const bool ring_exact =
      accel::device_vsm_ring_schedule_expected(
          test_case.proof->geometry, test_case.proof->width,
          expected_ring_reuse, expected_ring_checksum) &&
      accel::device_vsm_ring_storage_expected(
          test_case.proof->geometry, test_case.proof->width,
          test_case.proof->residents.count, expected_ring_state_bytes,
          expected_ring_scratch_bytes) &&
      accel::device_vsm_ring_round_trips_expected(
          test_case.proof->geometry, test_case.proof->residents.count,
          expected_ring_round_trips) &&
      evidence.ring_reuse_transitions == expected_ring_reuse &&
      evidence.ring_schedule_checksum == expected_ring_checksum &&
      evidence.ring_round_trips == expected_ring_round_trips &&
      evidence.ring_state_bytes == expected_ring_state_bytes &&
      evidence.ring_scratch_bytes == expected_ring_scratch_bytes;
  const bool valid = ring_exact && evidence.page_count == test_case.pages &&
                     evidence.generated_epochs == test_case.pages &&
                     evidence.completed_epochs == test_case.pages &&
                     evidence.forecasted_pages == test_case.pages &&
                     evidence.promoted_pages == test_case.pages &&
                     evidence.drained_pages == test_case.pages &&
                     evidence.persisted_pages == test_case.pages &&
                     evidence.native_submit_count == 1u &&
                     evidence.epoch_native_submit_count == 0u &&
                     evidence.payload_dispatch_count == 1u &&
                     evidence.host_service_turn_count == 0u &&
                     evidence.host_epoch_callback_count == 0u &&
                     evidence.final_callback_count == 1u;
  std::fprintf(
      stderr,
      "DeviceVsm binary backend=%u q=%llu valid=%u submit=%llu "
      "epoch_submit=%llu host_callback=%llu final=%llu ring=%llu/%llu/%llu "
      "retained=%llu\n",
      static_cast<unsigned>(test_case.backend),
      static_cast<unsigned long long>(test_case.pages),
      static_cast<unsigned>(valid),
      static_cast<unsigned long long>(evidence.native_submit_count),
      static_cast<unsigned long long>(evidence.epoch_native_submit_count),
      static_cast<unsigned long long>(evidence.host_epoch_callback_count),
      static_cast<unsigned long long>(evidence.final_callback_count),
      static_cast<unsigned long long>(evidence.ring_round_trips),
      static_cast<unsigned long long>(evidence.ring_state_bytes),
      static_cast<unsigned long long>(evidence.ring_scratch_bytes),
      static_cast<unsigned long long>(retained));
  return valid;
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test::binary_detail
