#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "evidence.hpp"
#include "fixture.hpp"
#include "route.hpp"

#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/run/sliding/internal.hpp"
#include "src/compute/virtual/state.hpp"

#include <algorithm>
#include <cstdio>

namespace rund_node_test_persistent_product {
namespace {

namespace accel = rund::node::accel::detail;
namespace sliding = rund::compute::detail::sliding_product_detail;

[[nodiscard]] bool
KnownAdmissionFinal(const ProductRouteObservation &observation,
                    const std::uint64_t coordinates) noexcept {
  if (!observation.production_route || !observation.final_received ||
      observation.final.check.ok ||
      observation.final.terminal != accel::NativeTerminal::Known) {
    return false;
  }
  const accel::PersistentResidencySlidingEvidence &evidence =
      observation.final.evidence;
  const std::uint64_t accepted = std::min<std::uint64_t>(2u, coordinates);
  return evidence.coordinate_count == coordinates &&
         evidence.accepted_coordinates == accepted &&
         evidence.accepted_end == accepted &&
         evidence.gpu_completed_coordinates == accepted &&
         evidence.completed_prefix == 0u && evidence.suppressed_first == 0u &&
         evidence.suppressed_count == accepted &&
         evidence.native_submit_count == 1u &&
         evidence.epoch_native_submit_count == 0u &&
         evidence.backend_epoch_callback_count == 0u &&
         evidence.host_epoch_callback_count == 0u &&
         evidence.backing_signal_count == accepted &&
         evidence.backing_wait_count == accepted &&
         evidence.backing_acknowledgement_count == accepted &&
         evidence.backing_service_failure_count == 0u &&
         evidence.failed_admission_count == accepted &&
         evidence.first_failed_admission_coordinate == 0u &&
         evidence.first_failure_coordinate == 0u &&
         evidence.final_callback_count == 1u && evidence.queue_calls == 1u &&
         evidence.chunk_submit_count == 1u;
}

[[nodiscard]] bool
KnownAuthorityAbort(const ProductRouteObservation &observation,
                    const std::uint64_t coordinates) noexcept {
  const auto owner =
      std::static_pointer_cast<sliding::SlidingProductOwner>(observation.owner);
  if (owner == nullptr || owner->run == nullptr) {
    return false;
  }
  rund::compute::detail::residency::execution::SlidingEvidence evidence{};
  const bool snapshot = owner->run->sliding.snapshot(evidence);
  const bool valid =
      snapshot && owner->run->completed && !owner->run->result.status &&
      !owner->run->result.poison_pipeline && !evidence.status &&
      evidence.planned == coordinates && evidence.admitted == 0u &&
      evidence.terminal_frontier == 0u && evidence.first_unsent == 0u &&
      evidence.has_failure && evidence.first_failure.ordinal == 0u &&
      evidence.first_failure_terminal ==
          rund::compute::detail::residency::execution::TerminalKind::Known &&
      evidence.terminal ==
          rund::compute::detail::residency::execution::TerminalKind::Known &&
      !evidence.quarantined && owner->run->sliding.quiescent();
  if (!valid) {
    std::fprintf(
        stderr,
        "persistent known authority snapshot=%u completed=%u status=%u "
        "poison=%u evidence=%u reason=%u plan=%llu admitted=%llu "
        "terminal=%llu unsent=%llu failure=%u@%llu first=%u whole=%u "
        "quarantine=%u quiescent=%u\n",
        static_cast<unsigned>(snapshot),
        static_cast<unsigned>(owner->run->completed),
        static_cast<unsigned>(bool(owner->run->result.status)),
        static_cast<unsigned>(owner->run->result.poison_pipeline),
        static_cast<unsigned>(bool(evidence.status)),
        static_cast<unsigned>(evidence.status.reason()),
        static_cast<unsigned long long>(evidence.planned),
        static_cast<unsigned long long>(evidence.admitted),
        static_cast<unsigned long long>(evidence.terminal_frontier),
        static_cast<unsigned long long>(evidence.first_unsent),
        static_cast<unsigned>(evidence.has_failure),
        static_cast<unsigned long long>(evidence.first_failure.ordinal),
        static_cast<unsigned>(evidence.first_failure_terminal),
        static_cast<unsigned>(evidence.terminal),
        static_cast<unsigned>(evidence.quarantined),
        static_cast<unsigned>(owner->run->sliding.quiescent()));
  }
  return valid;
}

} // namespace

bool CheckPersistentKnownAdmissionFailure(
    const rund::compute::Backend backend,
    const NativeQueueCounter queue_counter, bool &unavailable) noexcept {
  constexpr std::uint64_t coordinates = 5u;
  PreparedProduct prepared{};
  if (!PrepareProduct(backend, coordinates, prepared, unavailable)) {
    return unavailable;
  }
  const PublicationSnapshot primary_before =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot alternate_before =
      SnapshotPublication(prepared.state->alternate_pipeline);
  const std::uint64_t version_before = BackingVersion(*prepared.output);
  std::uint64_t queue_before = 0u;
  if (queue_counter == nullptr ||
      !queue_counter(prepared.state, queue_before)) {
    return false;
  }
  ProductRouteObservation failed{.fail_input = prepared.input.get()};
  const rund::compute::Status first =
      RunThroughPersistentProductRoute(prepared.state, failed);
  std::uint64_t queue_failed = 0u;
  if (first || !KnownAdmissionFinal(failed, coordinates) ||
      !KnownAuthorityAbort(failed, coordinates) ||
      !queue_counter(prepared.state, queue_failed) ||
      queue_failed != queue_before + 1u ||
      SnapshotPublication(prepared.state->pipeline).generation !=
          primary_before.generation ||
      SnapshotPublication(prepared.state->pipeline).payload_epoch !=
          primary_before.payload_epoch ||
      SnapshotPublication(prepared.state->alternate_pipeline).generation !=
          alternate_before.generation ||
      SnapshotPublication(prepared.state->alternate_pipeline).payload_epoch !=
          alternate_before.payload_epoch ||
      BackingVersion(*prepared.output) != version_before ||
      BackingRecovery(*prepared.output) != 0u ||
      prepared.input->read_batch_calls() != 1u ||
      prepared.input->read_batch_ranges() != ProductFrameCapacity) {
    const auto primary = SnapshotPublication(prepared.state->pipeline);
    const auto alternate =
        SnapshotPublication(prepared.state->alternate_pipeline);
    std::fprintf(
        stderr,
        "persistent known failure backend=%u status=%u final=%u/%u/%u "
        "gpu=%llu prefix=%llu suppress=%llu@%llu failed=%llu@%llu "
        "authority=%u queue=%llu/%llu bank=%llu/%llu,%llu/%llu "
        "backing=%llu/%llu recovery=%llu batch=%llu ranges=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(bool(first)),
        static_cast<unsigned>(failed.final_received),
        static_cast<unsigned>(failed.final.check.ok),
        static_cast<unsigned>(failed.final.terminal),
        static_cast<unsigned long long>(
            failed.final.evidence.gpu_completed_coordinates),
        static_cast<unsigned long long>(failed.final.evidence.completed_prefix),
        static_cast<unsigned long long>(failed.final.evidence.suppressed_count),
        static_cast<unsigned long long>(failed.final.evidence.suppressed_first),
        static_cast<unsigned long long>(
            failed.final.evidence.failed_admission_count),
        static_cast<unsigned long long>(
            failed.final.evidence.first_failed_admission_coordinate),
        static_cast<unsigned>(KnownAuthorityAbort(failed, coordinates)),
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_failed),
        static_cast<unsigned long long>(primary_before.generation),
        static_cast<unsigned long long>(primary.generation),
        static_cast<unsigned long long>(alternate_before.generation),
        static_cast<unsigned long long>(alternate.generation),
        static_cast<unsigned long long>(version_before),
        static_cast<unsigned long long>(BackingVersion(*prepared.output)),
        static_cast<unsigned long long>(BackingRecovery(*prepared.output)),
        static_cast<unsigned long long>(prepared.input->read_batch_calls()),
        static_cast<unsigned long long>(prepared.input->read_batch_ranges()));
    return false;
  }

  ProductRouteObservation retried{};
  const rund::compute::Status second =
      RunThroughPersistentProductRoute(prepared.state, retried);
  std::uint64_t queue_after = 0u;
  const bool valid =
      second && retried.production_route &&
      queue_counter(prepared.state, queue_after) &&
      queue_after == queue_before + 1u + PhysicalSubmits(coordinates) &&
      ExactOutput(prepared) &&
      ExactNativeFinal(retried, coordinates) &&
      ExactAuthorityFinal(retried, coordinates) &&
      SnapshotPublication(prepared.state->pipeline).generation ==
          primary_before.generation + 1u &&
      SnapshotPublication(prepared.state->alternate_pipeline).generation ==
          alternate_before.generation + 1u &&
      BackingVersion(*prepared.output) == version_before + 1u &&
      BackingRecovery(*prepared.output) == 0u &&
      prepared.input->read_batch_calls() == coordinates + 1u &&
      prepared.input->read_batch_ranges() ==
          ProductFrameCapacity + coordinates * ProductFrameCapacity - 1u;
  if (!valid) {
    std::fprintf(
        stderr,
        "persistent known retry backend=%u status=%u final=%u/%u "
        "queue=%llu/%llu output=%u native=%u authority=%u batch=%llu "
        "ranges=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(bool(second)),
        static_cast<unsigned>(retried.final_received),
        static_cast<unsigned>(retried.final.check.ok),
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_after),
        static_cast<unsigned>(ExactOutput(prepared)),
        static_cast<unsigned>(ExactNativeFinal(retried, coordinates)),
        static_cast<unsigned>(ExactAuthorityFinal(retried, coordinates)),
        static_cast<unsigned long long>(prepared.input->read_batch_calls()),
        static_cast<unsigned long long>(prepared.input->read_batch_ranges()));
  }
  return valid;
}

} // namespace rund_node_test_persistent_product

#endif
