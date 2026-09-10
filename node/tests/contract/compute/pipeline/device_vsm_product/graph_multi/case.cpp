#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../persistent_product/fixture.hpp"
#include "../evidence.hpp"

#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"
#include <array>
#include <cstdio>

namespace rund_node_test_device_vsm_product::graph_multi_test {
namespace {

bool RunGraphMultiCase(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    const std::uint64_t pages, const rund::kernel::ReduceOp operation,
    bool &unavailable) noexcept {
  using namespace rund_node_test_persistent_product;
  PreparedGraphMulti prepared{};
  if (!PrepareGraphMulti(backend, pages, operation, prepared, unavailable)) {
    return unavailable;
  }
  const PublicationSnapshot before_map =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot before_reduce = SnapshotPublication(
      rund::compute::detail::graph_terminal_pipeline(*prepared.state, 0u));
  const std::uint64_t version_before = BackingVersion(*prepared.output);
  std::uint64_t queue_before = 0u;
  if (queue_counter == nullptr ||
      !queue_counter(prepared.state, queue_before)) {
    return false;
  }
  RouteObservation observation{};
  const rund::compute::Status status =
      RunThroughDeviceVsmProductRoute(prepared.state, observation);
  std::uint64_t queue_after = 0u;
  const PublicationSnapshot after_map =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot after_reduce = SnapshotPublication(
      rund::compute::detail::graph_terminal_pipeline(*prepared.state, 0u));
  const auto owner = std::static_pointer_cast<
      rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner>(
      observation.owner);
  const rund::compute::ResidencyStats &stats =
      prepared.state->stats.pipeline.residency;
  const bool queue_valid = queue_counter(prepared.state, queue_after) &&
                           queue_after == queue_before + 1u;
  const bool output_valid = ExactGraphMultiOutput(prepared);
  const bool proof_valid =
      owner != nullptr && owner->proof != nullptr &&
      owner->proof->topology ==
          rund::node::accel::detail::DeviceVsmTopology::GraphMapReduce &&
      owner->proof->graph_map_reduce.semantic.op == operation &&
      owner->proof->plan.input_buffer_count == 2u &&
      owner->proof->plan.output_buffer_count == 1u &&
      owner->proof->residents.input_count == 2u &&
      owner->proof->residents.output_count == 1u;
  const bool evidence_valid =
      ExactDeviceVsmEvidence(observation, pages, 1u, 2u * prepared.input_bytes,
                             sizeof(std::uint64_t), 0u);
  const bool publication_valid = ExactPublication(
      before_map, after_map, before_reduce, after_reduce, version_before,
      BackingVersion(*prepared.output), BackingRecovery(*prepared.output));
  const bool stats_valid =
      stats.window_handoff_count == 1u && stats.window_batch_count == 1u &&
      stats.window_queue_call_count == 1u &&
      stats.page_in_count == 2u * pages && stats.page_out_count == 1u &&
      stats.backing_read_bytes == 2u * prepared.input_bytes &&
      stats.backing_write_bytes == sizeof(std::uint64_t) &&
      prepared.state->stats.host_write_bytes == 2u * prepared.input_bytes &&
      prepared.state->stats.uploaded_bytes == 0u &&
      prepared.state->stats.downloaded_bytes == 0u &&
      prepared.state->stats.transfer_submissions.host_to_device == 0u &&
      prepared.state->stats.transfer_submissions.device_to_host == 0u;
  const bool valid = status && queue_valid && output_valid && proof_valid &&
                     evidence_valid && publication_valid && stats_valid;
  if (!valid) {
    std::fprintf(
        stderr,
        "Graph multi contracts queue=%u output=%u proof=%u evidence=%u "
        "publication=%u stats=%u queue_count=%llu/%llu\n",
        queue_valid, output_valid, proof_valid, evidence_valid,
        publication_valid, stats_valid,
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_after));
  }
  std::fprintf(
      stderr,
      "DeviceVsm Graph multi backend=%u op=%u Q=%llu valid=%u submit=%llu "
      "epoch_submit=%llu host_turn=%llu host_callback=%llu final=%llu "
      "inputs=%u read=%llu publish=%llu/%llu/%llu status=%u reason=%u "
      "route=%u/%u/%u/%s\n",
      static_cast<unsigned>(backend), static_cast<unsigned>(operation),
      static_cast<unsigned long long>(pages), static_cast<unsigned>(valid),
      static_cast<unsigned long long>(
          observation.evidence.native.native_submit_count),
      static_cast<unsigned long long>(
          observation.evidence.native.epoch_native_submit_count),
      static_cast<unsigned long long>(
          observation.evidence.native.host_service_turn_count),
      static_cast<unsigned long long>(
          observation.evidence.native.host_epoch_callback_count),
      static_cast<unsigned long long>(
          observation.evidence.native.final_callback_count),
      owner == nullptr || owner->proof == nullptr
          ? 0u
          : owner->proof->residents.input_count,
      static_cast<unsigned long long>(stats.backing_read_bytes),
      static_cast<unsigned long long>(
          observation.evidence.authority_accept_count),
      static_cast<unsigned long long>(
          observation.evidence.pipeline_terminal_count),
      static_cast<unsigned long long>(
          observation.evidence.backing_publication_count),
      static_cast<unsigned>(static_cast<bool>(status)),
      static_cast<unsigned>(status.reason()),
      static_cast<unsigned>(observation.production_route),
      static_cast<unsigned>(observation.prepared),
      static_cast<unsigned>(observation.executed),
      observation.prepare_reason == nullptr ? "null"
                                            : observation.prepare_reason);
  return valid;
}

} // namespace

bool RunGraphMultiProductCases(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter
        queue_counter) noexcept {
  for (const rund::kernel::ReduceOp operation :
       {rund::kernel::ReduceOp::Sum, rund::kernel::ReduceOp::CountNonzero,
        rund::kernel::ReduceOp::Min, rund::kernel::ReduceOp::Max}) {
    for (const std::uint64_t pages : std::array{5u, 9u, 257u}) {
      bool unavailable = false;
      if (!RunGraphMultiCase(backend, queue_counter, pages, operation,
                             unavailable)) {
        return false;
      }
      if (unavailable) {
        return true;
      }
    }
  }
  return RunGraphMaximumInputProduct(backend, queue_counter);
}

} // namespace rund_node_test_device_vsm_product::graph_multi_test

#endif
