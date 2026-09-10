#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "evidence.hpp"
#include "fixture.hpp"
#include "route.hpp"

#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/run/sliding/internal.hpp"
#include "src/compute/virtual/state.hpp"

#include <cstdio>

namespace rund_node_test_persistent_product {
namespace {

[[nodiscard]] bool Unchanged(const PublicationSnapshot before,
                             const PublicationSnapshot after) noexcept {
  return before.generation == after.generation &&
         before.payload_epoch == after.payload_epoch;
}

} // namespace

bool CheckPersistentPublicationFailure(const rund::compute::Backend backend,
                                       const NativeQueueCounter queue_counter,
                                       bool &unavailable) noexcept {
  PreparedProduct prepared{};
  if (!PrepareProduct(backend, 9u, prepared, unavailable)) {
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

  rund::compute::detail::sliding_product_detail::
      inject_persistent_publication_preflight_failure_once();
  ProductRouteObservation rejected{};
  const rund::compute::Status first =
      RunThroughPersistentProductRoute(prepared.state, rejected);
  const PublicationSnapshot primary_rejected =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot alternate_rejected =
      SnapshotPublication(prepared.state->alternate_pipeline);
  const std::uint64_t version_rejected = BackingVersion(*prepared.output);
  const std::uint64_t recovery_rejected = BackingRecovery(*prepared.output);
  std::uint64_t queue_rejected = 0u;
  const std::uint64_t logical_output_bytes =
      prepared.expected.size() * sizeof(std::uint32_t);
  const std::uint64_t submits = PhysicalSubmits(9u);
  const bool first_mode =
      ExactMode(rejected,
                rund::node::accel::detail::PersistentResidencySlidingMode::
                    BackendChunked);
  const bool first_native = ExactNativeFinal(rejected, 9u);
  const bool first_authority = ExactAuthorityFinal(rejected, 9u);
  const bool first_valid =
      !first && rejected.production_route && rejected.final_received &&
      rejected.final.check.ok && first_mode && first_native &&
      Unchanged(primary_before, primary_rejected) &&
      Unchanged(alternate_before, alternate_rejected) &&
      version_rejected == version_before &&
      recovery_rejected == logical_output_bytes &&
      queue_counter(prepared.state, queue_rejected) &&
      queue_rejected == queue_before + submits;
  if (!first_valid) {
    std::fprintf(
        stderr,
        "persistent publication preflight backend=%u status=%u reason=%u "
        "queue=%llu/%llu/0 mode=%u native=%u authority=%u "
        "pub=%llu/%llu version=%llu/%llu recovery=%llu/%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(bool(first)),
        static_cast<unsigned>(first.reason()),
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_rejected),
        static_cast<unsigned>(first_mode), static_cast<unsigned>(first_native),
        static_cast<unsigned>(first_authority),
        static_cast<unsigned long long>(primary_rejected.generation),
        static_cast<unsigned long long>(alternate_rejected.generation),
        static_cast<unsigned long long>(version_before),
        static_cast<unsigned long long>(version_rejected),
        static_cast<unsigned long long>(recovery_rejected),
        static_cast<unsigned long long>(logical_output_bytes));
    return false;
  }

  ProductRouteObservation retried{};
  const rund::compute::Status second =
      RunThroughPersistentProductRoute(prepared.state, retried);
  const PublicationSnapshot primary_after =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot alternate_after =
      SnapshotPublication(prepared.state->alternate_pipeline);
  std::uint64_t queue_after = 0u;
  const std::uint64_t version_after = BackingVersion(*prepared.output);
  const std::uint64_t recovery_after = BackingRecovery(*prepared.output);
  const bool retry_mode =
      ExactMode(retried,
                rund::node::accel::detail::PersistentResidencySlidingMode::
                    BackendChunked);
  const bool retry_native = ExactNativeFinal(retried, 9u);
  const bool retry_authority = ExactAuthorityFinal(retried, 9u);
  const bool valid =
      second && retried.production_route && retried.final_received &&
      retried.final.check.ok && retry_mode && retry_native &&
      retry_authority && queue_counter(prepared.state, queue_after) &&
      queue_after == queue_before + submits * 2u && ExactOutput(prepared) &&
      primary_after.generation == primary_before.generation + 1u &&
      primary_after.payload_epoch == primary_before.payload_epoch + 1u &&
      alternate_after.generation == alternate_before.generation + 1u &&
      alternate_after.payload_epoch == alternate_before.payload_epoch + 1u &&
      version_after == version_before + 1u && recovery_after == 0u;
  if (!valid) {
    std::fprintf(
        stderr,
        "persistent publication retry backend=%u status=%u reason=%u "
        "queue=%llu/%llu/%llu mode=%u native=%u authority=%u "
        "pub=%llu/%llu version=%llu/%llu recovery=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(bool(second)),
        static_cast<unsigned>(second.reason()),
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_rejected),
        static_cast<unsigned long long>(queue_after),
        static_cast<unsigned>(retry_mode), static_cast<unsigned>(retry_native),
        static_cast<unsigned>(retry_authority),
        static_cast<unsigned long long>(primary_after.generation),
        static_cast<unsigned long long>(alternate_after.generation),
        static_cast<unsigned long long>(version_before),
        static_cast<unsigned long long>(version_after),
        static_cast<unsigned long long>(recovery_after));
  }
  return valid;
}

} // namespace rund_node_test_persistent_product

#endif
