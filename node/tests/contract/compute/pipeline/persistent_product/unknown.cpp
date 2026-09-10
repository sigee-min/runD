#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "fixture.hpp"
#include "route.hpp"

#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/run/sliding/persistent/service/internal.hpp"
#include "src/compute/virtual/state.hpp"

#include <algorithm>
#include <cstdio>

namespace rund_node_test_persistent_product {
namespace {

namespace accel = rund::node::accel::detail;
namespace sliding = rund::compute::detail::sliding_product_detail;

[[nodiscard]] bool UnknownFinal(const ProductRouteObservation &observation,
                                const std::uint64_t coordinates) noexcept {
  if (!observation.production_route || !observation.final_received ||
      observation.final.check.ok ||
      observation.final.terminal != accel::NativeTerminal::UnknownMayWrite) {
    return false;
  }
  const accel::PersistentResidencySlidingEvidence &evidence =
      observation.final.evidence;
  const std::uint64_t accepted = std::min<std::uint64_t>(2u, coordinates);
  return evidence.coordinate_count == coordinates &&
         evidence.accepted_coordinates == accepted &&
         evidence.accepted_end == accepted &&
         evidence.native_submit_count == 1u &&
         evidence.epoch_native_submit_count == 0u &&
         evidence.backend_epoch_callback_count == 0u &&
         evidence.host_epoch_callback_count == 0u &&
         evidence.backing_signal_count == 0u &&
         evidence.backing_wait_count == 0u &&
         evidence.backing_acknowledgement_count == 0u &&
         evidence.backing_service_failure_count == 1u &&
         evidence.first_service_failure_coordinate == 0u &&
         evidence.failed_admission_count == 0u &&
         evidence.first_failure_coordinate == 0u &&
         evidence.final_callback_count == 1u && evidence.queue_calls == 1u &&
         evidence.chunk_submit_count == 1u;
}

[[nodiscard]] bool
QuarantinedLifetime(const ProductRouteObservation &observation) noexcept {
  const auto owner =
      std::static_pointer_cast<sliding::SlidingProductOwner>(observation.owner);
  if (owner == nullptr || owner->run == nullptr ||
      owner->run->quarantine.get() != owner->run.get() ||
      !owner->run->completed || owner->run->result.status ||
      !owner->run->result.poison_pipeline ||
      !owner->run->poison.load(std::memory_order_acquire) ||
      !owner->run->sliding.quarantined()) {
    return false;
  }
  std::lock_guard lock{owner->run->persistent_control.gate};
  return !owner->run->persistent_control.active &&
         owner->run->persistent_control.quarantined &&
         owner->run->persistent_control.native != nullptr;
}

} // namespace

bool CheckPersistentUnknownFailure(const rund::compute::Backend backend,
                                   const NativeQueueCounter queue_counter,
                                   bool &unavailable) noexcept {
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

  sliding::inject_persistent_service_unknown_once();
  ProductRouteObservation observation{};
  const rund::compute::Status first =
      RunThroughPersistentProductRoute(prepared.state, observation);
  std::uint64_t queue_failed = 0u;
  const bool failed =
      !first && UnknownFinal(observation, coordinates) &&
      QuarantinedLifetime(observation) &&
      queue_counter(prepared.state, queue_failed) &&
      queue_failed == queue_before + 1u &&
      SnapshotPublication(prepared.state->pipeline).generation ==
          primary_before.generation &&
      SnapshotPublication(prepared.state->pipeline).payload_epoch ==
          primary_before.payload_epoch &&
      SnapshotPublication(prepared.state->alternate_pipeline).generation ==
          alternate_before.generation &&
      SnapshotPublication(prepared.state->alternate_pipeline).payload_epoch ==
          alternate_before.payload_epoch &&
      BackingVersion(*prepared.output) == version_before &&
      BackingRecovery(*prepared.output) == 0u;

  ProductRouteObservation retried{};
  const rund::compute::Status retry =
      RunThroughPersistentProductRoute(prepared.state, retried);
  std::uint64_t queue_after = 0u;
  const bool valid =
      failed && !retry && retried.production_route && !retried.final_received &&
      queue_counter(prepared.state, queue_after) &&
      queue_after == queue_failed && QuarantinedLifetime(observation);
  if (!valid) {
    std::fprintf(
        stderr,
        "persistent unknown backend=%u status=%u retry=%u final=%u/%u/%u "
        "submit=%llu service=%llu queue=%llu/%llu/%llu lifetime=%u "
        "backing=%llu/%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(bool(first)),
        static_cast<unsigned>(bool(retry)),
        static_cast<unsigned>(observation.final_received),
        static_cast<unsigned>(observation.final.check.ok),
        static_cast<unsigned>(observation.final.terminal),
        static_cast<unsigned long long>(
            observation.final.evidence.native_submit_count),
        static_cast<unsigned long long>(
            observation.final.evidence.backing_service_failure_count),
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_failed),
        static_cast<unsigned long long>(queue_after),
        static_cast<unsigned>(QuarantinedLifetime(observation)),
        static_cast<unsigned long long>(version_before),
        static_cast<unsigned long long>(BackingVersion(*prepared.output)));
  }
  return valid;
}

} // namespace rund_node_test_persistent_product

#endif
