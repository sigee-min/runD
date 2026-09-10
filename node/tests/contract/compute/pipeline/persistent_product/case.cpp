#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "evidence.hpp"
#include "fixture.hpp"
#include "route.hpp"

#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/state.hpp"

namespace rund_node_test_persistent_product {

bool RunPersistentProductCase(const rund::compute::Backend backend,
                              const NativeQueueCounter queue_counter,
                              const std::uint64_t coordinates,
                              bool &unavailable) noexcept {
  PreparedProduct prepared{};
  if (!PrepareProduct(backend, coordinates, prepared, unavailable)) {
    return unavailable;
  }
  const PublicationSnapshot before_primary =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot before_alternate =
      SnapshotPublication(prepared.state->alternate_pipeline);
  const std::uint64_t version_before = BackingVersion(*prepared.output);
  std::uint64_t queue_before = 0u;
  if (queue_counter == nullptr ||
      !queue_counter(prepared.state, queue_before)) {
    return false;
  }
  const std::uint64_t pages = coordinates * ProductFrameCapacity - 1u;
  const std::uint64_t backing_input_bytes =
      (pages * ProductPageElements - 3u) * sizeof(std::uint32_t);
  const std::uint64_t materialized_input_bytes =
      pages * ProductPageElements * sizeof(std::uint32_t);
  ProductRouteObservation observation{};
  const rund::compute::Status status =
      RunThroughPersistentProductRoute(prepared.state, observation);
  std::uint64_t queue_after = 0u;
  const bool queue_read = queue_counter(prepared.state, queue_after);
  RunSample sample{};
  sample.backend = backend;
  sample.coordinates = coordinates;
  sample.status = status;
  sample.observation = observation;
  sample.product = &prepared;
  sample.queue_before = queue_before;
  sample.queue_after = queue_after;
  sample.command_submits = prepared.state->stats.command_submits;
  sample.queue_read = queue_read;
  sample.before_primary = before_primary;
  sample.after_primary = SnapshotPublication(prepared.state->pipeline);
  sample.before_alternate = before_alternate;
  sample.after_alternate =
      SnapshotPublication(prepared.state->alternate_pipeline);
  sample.version_before = version_before;
  sample.version_after = BackingVersion(*prepared.output);
  sample.recovery_after = BackingRecovery(*prepared.output);
  sample.stats = prepared.state->stats.pipeline.residency;
  sample.input_read_calls = prepared.input->read_batch_calls();
  sample.input_read_ranges = prepared.input->read_batch_ranges();
  sample.pages = pages;
  sample.backing_input_bytes = backing_input_bytes;
  sample.materialized_input_bytes = materialized_input_bytes;
  return ExactRun(sample);
}

} // namespace rund_node_test_persistent_product

#endif
