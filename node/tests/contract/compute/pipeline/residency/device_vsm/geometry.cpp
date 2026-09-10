#include "local.hpp"

#include <limits>

namespace rund_node_test_pipeline_residency::device_vsm_test {

bool CheckGeometry() noexcept {
  const auto proof = Proof(5u);
  accel::DeviceVsmPageProjection first{};
  accel::DeviceVsmPageProjection middle{};
  accel::DeviceVsmPageProjection last{};
  if (!accel::device_vsm_proof_valid(*proof) ||
      !accel::device_vsm_project_page(proof->geometry, 0u, first) ||
      !accel::device_vsm_project_page(proof->geometry, 2u, middle) ||
      !accel::device_vsm_project_page(proof->geometry, 4u, last) ||
      first.core_offset != 0u || first.core_bytes != 64u ||
      first.source_offset != 0u || first.source_bytes != 80u ||
      first.frame_source_offset != 16u || middle.core_offset != 128u ||
      middle.source_offset != 112u || middle.source_bytes != 96u ||
      middle.frame_source_offset != 0u || last.core_offset != 256u ||
      last.core_bytes != 52u || last.source_offset != 240u ||
      last.source_bytes != 68u || last.frame_source_offset != 0u) {
    return false;
  }
  std::uint64_t overlap = 0u;
  std::uint32_t ring_reuse = 0u;
  std::uint32_t ring_checksum = 0u;
  std::uint32_t ring_round_trips = 0u;
  std::uint64_t ring_state_bytes = 0u;
  std::uint64_t ring_scratch_bytes = 0u;
  accel::DeviceVsmWindowFootprintAuthority footprint{};
  if (!accel::device_vsm_overlap_reuse_bytes(proof->geometry, overlap) ||
      overlap != 128u ||
      !accel::device_vsm_ring_schedule_expected(proof->geometry, 2u, ring_reuse,
                                                ring_checksum) ||
      ring_reuse != 3u || ring_checksum != 990u ||
      !accel::device_vsm_ring_storage_expected(
          proof->geometry, 2u, 2u, ring_state_bytes, ring_scratch_bytes) ||
      !accel::device_vsm_ring_round_trips_expected(proof->geometry, 2u,
                                                   ring_round_trips) ||
      ring_state_bytes != 8u || ring_scratch_bytes != 256u ||
      ring_round_trips != 10u ||
      !accel::device_vsm_project_window_footprint(proof->geometry, footprint) ||
      footprint != proof->window.footprint ||
      footprint.canonical_page_bytes != 64u ||
      footprint.boundary_transition_count != 4u ||
      footprint.projection_checksum != 1020u) {
    return false;
  }
  if (accel::device_vsm_ring_schedule_expected(proof->geometry, 1u, ring_reuse,
                                               ring_checksum) ||
      ring_reuse != 0u || ring_checksum != 0u ||
      accel::device_vsm_ring_storage_expected(
          proof->geometry, 1u, 2u, ring_state_bytes, ring_scratch_bytes) ||
      ring_state_bytes != 0u || ring_scratch_bytes != 0u) {
    return false;
  }
  auto malformed = proof->geometry;
  malformed.page_count = 4u;
  if (accel::device_vsm_geometry_valid(malformed)) {
    return false;
  }
  malformed = proof->geometry;
  malformed.target_offset_bytes = 88u;
  if (accel::device_vsm_geometry_valid(malformed)) {
    return false;
  }

  auto wrong_artifact = *proof;
  auto artifact = *proof->artifact;
  artifact.key.variant = rund::kernel::LoweringArtifactVariant::Recurrence;
  wrong_artifact.artifact = &artifact;
  if (accel::device_vsm_proof_valid(wrong_artifact)) {
    return false;
  }

  auto wrong_footprint = *proof;
  ++wrong_footprint.window.footprint.projection_checksum;
  if (accel::device_vsm_proof_valid(wrong_footprint)) {
    return false;
  }

  malformed = proof->geometry;
  constexpr std::uint64_t TooManyElements =
      std::uint64_t{std::numeric_limits<std::uint32_t>::max()} + 1u;
  malformed.logical_bytes = TooManyElements * malformed.element_bytes;
  malformed.payload_bytes = (TooManyElements / 2u) * malformed.element_bytes;
  malformed.frame_bytes = malformed.payload_bytes + 32u;
  malformed.page_count = 2u;
  return accel::device_vsm_geometry_valid(malformed) &&
         !accel::device_vsm_runtime_geometry_valid(malformed);
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test
