#pragma once

#include "internal.hpp"

#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund_node_test_virtual::product::graph_resident::oracle_detail {

using Owner =
    rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;

[[nodiscard]] inline std::shared_ptr<Owner>
owner_of(const Case &test_case) noexcept {
  return test_case.state == nullptr
             ? std::shared_ptr<Owner>{}
             : std::static_pointer_cast<Owner>(
                   test_case.state->device_vsm_product_cache);
}

[[nodiscard]] inline std::shared_ptr<Owner>
owner_of(const U32Case &test_case) noexcept {
  return test_case.state == nullptr
             ? std::shared_ptr<Owner>{}
             : std::static_pointer_cast<Owner>(
                   test_case.state->device_vsm_product_cache);
}

[[nodiscard]] inline bool time_ok(const Observation &observation) noexcept {
  const auto &after = observation.after;
  return observation.native_kernel_ns == after.kernel_ns &&
         observation.native_kernel_samples == after.kernel_samples &&
         observation.native_submit_wait_ns == after.submit_wait_ns &&
         (observation.native_kernel_samples != 0u ||
          observation.native_kernel_ns == 0u);
}

[[nodiscard]] inline bool
completion_ok(const Observation &observation) noexcept {
  return observation.native_submit_count == 0u ||
         observation.native_completed_ns != 0u;
}

[[nodiscard]] inline bool route_ok(const Observation &observation) noexcept {
  return observation.status && observation.graph_resident &&
         observation.route_kind == RouteKind::DeviceVsm &&
         observation.accepted_owner_mask == OwnerDeviceVsm &&
         observation.accepted_owner_count == 1u;
}

[[nodiscard]] inline bool
native_ok(const Observation &observation, const std::uint64_t page_count,
          const std::uint64_t element_count,
          const std::uint64_t element_bytes = sizeof(std::uint64_t)) noexcept {
  const std::uint64_t output_bytes =
      element_count * element_bytes;
  return observation.native_submit_count == 1u &&
         observation.epoch_submit_count == 0u &&
         observation.dispatch_count == 1u &&
         observation.final_count == 1u &&
         observation.tile_dispatch_count == 1u &&
         observation.generated_pages == page_count &&
         observation.completed_pages == page_count &&
         observation.forecasted_pages == page_count &&
         observation.promoted_pages == page_count &&
         observation.drained_pages == page_count &&
         observation.persisted_pages == page_count &&
         observation.wavefront_steps == StageCount * BatchCount &&
         observation.gpu_read_bytes == 0u &&
         observation.gpu_write_bytes == output_bytes &&
         observation.native_may_write && !observation.host_service &&
         !observation.quarantined && observation.resource_count == 8u &&
         observation.internal_owners == InternalOwnerCount &&
         observation.proof_digest != 0u &&
         observation.native_proof_hi != 0u &&
         observation.native_proof_lo != 0u &&
         observation.native_generation != 0u &&
         observation.native_nonce != 0u;
}

[[nodiscard]] inline bool
transfer_ok(const Observation &observation,
            const std::uint64_t element_count,
            const std::uint64_t element_bytes = sizeof(std::uint64_t)) noexcept {
  const auto &residency = observation.after.pipeline.residency;
  const std::uint64_t output_bytes = element_count * element_bytes;
  return observation.staged_output
             ? residency.backing_read_bytes ==
                   static_cast<std::uint64_t>(InputCount) * output_bytes &&
                   residency.backing_write_bytes == output_bytes
             : residency.backing_read_bytes == 0u &&
                   residency.backing_write_bytes == 0u;
}

[[nodiscard]] inline bool
stats_ok(const Observation &observation, const std::uint64_t page_count,
         const std::uint64_t element_count,
         const std::uint64_t element_bytes = sizeof(std::uint64_t)) noexcept {
  const auto &after = observation.after;
  const auto &residency = after.pipeline.residency;
  const std::uint64_t output_bytes = element_count * element_bytes;
  const bool transfer = transfer_ok(observation, element_count, element_bytes);
  return after.command_submits == 1u && after.dispatches == 1u &&
         after.final_dispatches == 1u &&
         (observation.staged_output
              ? transfer
              : after.uploaded_bytes == 0u && after.downloaded_bytes == 0u &&
                    after.external_roundtrip_bytes == 0u && transfer) &&
         residency.epoch_count == BatchCount &&
         residency.page_in_count ==
             static_cast<std::uint64_t>(InputCount) * page_count &&
         residency.page_in_bytes == 0u &&
         residency.page_out_bytes == output_bytes &&
         residency.frame_capacity == 2u;
}

[[nodiscard]] inline bool
output_ok(const Observation &observation,
          const std::uint64_t expected_hash) noexcept {
  return observation.output_match && observation.output_hash == expected_hash;
}

struct Continuity final {
  bool version{};
  bool proof{};
  bool type{};
  bool generation{};
  bool nonce{};

  [[nodiscard]] bool all() const noexcept {
    return version && proof && type && generation && nonce;
  }
};

[[nodiscard]] inline Continuity
continuity(const std::span<const Observation> observations) noexcept {
  Continuity result{.version = !observations.empty(),
                    .proof = !observations.empty(),
                    .type = !observations.empty(),
                    .generation = !observations.empty(),
                    .nonce = !observations.empty()};
  for (std::size_t index = 1u; index < observations.size(); ++index) {
    const Observation &prior = observations[index - 1u];
    const Observation &current = observations[index];
    result.version =
        result.version && current.version_before == prior.version_after;
    result.proof = result.proof && current.proof_digest == prior.proof_digest &&
                   current.native_proof_hi == prior.native_proof_hi &&
                   current.native_proof_lo == prior.native_proof_lo;
    result.type = result.type &&
                  current.proof_scalar == prior.proof_scalar &&
                  current.proof_domain == prior.proof_domain &&
                  current.proof_element_bytes == prior.proof_element_bytes;
    result.generation = result.generation &&
                        current.native_generation > prior.native_generation;
    result.nonce = result.nonce && current.native_nonce != prior.native_nonce;
  }
  return result;
}

} // namespace rund_node_test_virtual::product::graph_resident::oracle_detail
