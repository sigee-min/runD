#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "binary/local.hpp"
#include "evidence.hpp"
#include "graph/local.hpp"
#include "graph_multi/local.hpp"
#include "multi_scan/local.hpp"
#include "pointwise_dag/local.hpp"
#include "reduce_ops/local.hpp"
#include "reduce_sum/local.hpp"
#include "route.hpp"
#include "scan/local.hpp"
#include "scan_u32/local.hpp"
#include "window/local.hpp"

#include "../persistent_product/fixture.hpp"

#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/state.hpp"

#include <cstdio>

namespace rund_node_test_device_vsm_product {
namespace {

bool RunCase(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    const std::uint64_t pages, bool &unavailable,
    const bool force_device_vsm = true) noexcept {
  using namespace rund_node_test_persistent_product;
  PreparedProduct prepared{};
  const std::uint64_t sliding_coordinates = (pages + 1u) / 2u;
  if (!PrepareProduct(backend, sliding_coordinates, prepared, unavailable)) {
    return unavailable;
  }
  if (prepared.state->pipeline->residency->stream().page_count() != pages) {
    return false;
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
  RouteObservation observation{};
  const rund::compute::Status status =
      RunThroughDeviceVsmProductRoute(prepared.state, observation,
                                      force_device_vsm);
  const auto owner = std::static_pointer_cast<
      rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner>(
      observation.owner);
  std::uint64_t queue_after = 0u;
  const bool queue_exact = queue_counter(prepared.state, queue_after) &&
                           queue_after == queue_before + 1u;
  const PublicationSnapshot after_primary =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot after_alternate =
      SnapshotPublication(prepared.state->alternate_pipeline);
  const std::uint64_t version_after = BackingVersion(*prepared.output);
  const std::uint64_t recovery_after = BackingRecovery(*prepared.output);
  const std::uint64_t bytes = prepared.expected.size() * sizeof(std::uint32_t);
  const rund::compute::Stats &run_stats = prepared.state->stats;
  const rund::compute::ResidencyStats &stats =
      prepared.state->stats.pipeline.residency;
  const bool valid =
      status && queue_exact && ExactOutput(prepared) && owner != nullptr &&
      owner->input_count == 1u && owner->evidence != nullptr &&
      owner->evidence->public_resident_input_count == 0u &&
      owner->evidence->whole_run_staged_input_count == 1u &&
      !owner->evidence->public_resident_output &&
      owner->evidence->whole_run_staged_output &&
      !owner->evidence->bounded_external_page_service &&
      (force_device_vsm ||
       (owner != nullptr &&
        owner->route_proof.kind() ==
            rund::compute::detail::VirtualDeviceVsmRouteKind::StagedLoop)) &&
      ExactDeviceVsmEvidence(observation, pages, pages, bytes, bytes, 0u) &&
      ExactPublication(before_primary, after_primary, before_alternate,
                       after_alternate, version_before, version_after,
                       recovery_after) &&
      stats.window_handoff_count == 1u && stats.window_batch_count == 1u &&
      stats.window_queue_call_count == 1u && run_stats.command_submits == 1u &&
      ExactWholeRunTransfers(*owner, run_stats, bytes) &&
      stats.page_in_count == pages && stats.page_out_count == pages &&
      stats.backing_read_bytes == bytes && stats.backing_write_bytes == bytes &&
      stats.page_in_bytes == bytes && stats.page_out_bytes == bytes;
  if (valid) {
    std::fprintf(
        stderr,
        "DeviceVsm product backend=%u Q=%llu submit=%llu epoch_submit=%llu "
        "host_turn=%llu host_callback=%llu generated=%llu completed=%llu "
        "final=%u authority=%llu pipeline=%llu backing=%llu "
        "public=%u/%u staged=%u/%u external=%u\n",
        static_cast<unsigned>(backend), static_cast<unsigned long long>(pages),
        static_cast<unsigned long long>(
            observation.evidence.native.native_submit_count),
        static_cast<unsigned long long>(
            observation.evidence.native.epoch_native_submit_count),
        static_cast<unsigned long long>(
            observation.evidence.native.host_service_turn_count),
        static_cast<unsigned long long>(
            observation.evidence.native.host_epoch_callback_count),
        static_cast<unsigned long long>(
            observation.evidence.native.generated_epochs),
        static_cast<unsigned long long>(
            observation.evidence.native.completed_epochs),
        static_cast<unsigned>(observation.evidence.final_received),
        static_cast<unsigned long long>(
            observation.evidence.authority_accept_count),
        static_cast<unsigned long long>(
            observation.evidence.pipeline_terminal_count),
        static_cast<unsigned long long>(
            observation.evidence.backing_publication_count),
        owner == nullptr || owner->evidence == nullptr
            ? 0u
            : owner->evidence->public_resident_input_count,
        static_cast<unsigned>(owner != nullptr && owner->evidence != nullptr &&
                              owner->evidence->public_resident_output),
        owner == nullptr || owner->evidence == nullptr
            ? 0u
            : owner->evidence->whole_run_staged_input_count,
        static_cast<unsigned>(owner != nullptr && owner->evidence != nullptr &&
                              owner->evidence->whole_run_staged_output),
        static_cast<unsigned>(owner != nullptr && owner->evidence != nullptr &&
                              owner->evidence->bounded_external_page_service));
  }
  if (!valid) {
    std::fprintf(
        stderr,
        "DeviceVsm product backend=%u Q=%llu status=%u route=%u/%u/%u/%s "
        "queue=%llu/%llu final=%u generated=%llu completed=%llu "
        "submit=%llu epoch_submit=%llu host_turn=%llu host_callback=%llu "
        "publish=%llu/%llu/%llu/%llu backing=%llu/%llu recovery=%llu "
        "stats=%llu/%llu/%llu/%llu/%llu output=%u\n",
        static_cast<unsigned>(backend), static_cast<unsigned long long>(pages),
        static_cast<unsigned>(static_cast<bool>(status)),
        static_cast<unsigned>(observation.production_route),
        static_cast<unsigned>(observation.prepared),
        static_cast<unsigned>(observation.executed),
        observation.prepare_reason == nullptr ? "null"
                                              : observation.prepare_reason,
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_after),
        static_cast<unsigned>(observation.evidence.final_received),
        static_cast<unsigned long long>(
            observation.evidence.native.generated_epochs),
        static_cast<unsigned long long>(
            observation.evidence.native.completed_epochs),
        static_cast<unsigned long long>(
            observation.evidence.native.native_submit_count),
        static_cast<unsigned long long>(
            observation.evidence.native.epoch_native_submit_count),
        static_cast<unsigned long long>(
            observation.evidence.native.host_service_turn_count),
        static_cast<unsigned long long>(
            observation.evidence.native.host_epoch_callback_count),
        static_cast<unsigned long long>(before_primary.generation),
        static_cast<unsigned long long>(after_primary.generation),
        static_cast<unsigned long long>(before_alternate.generation),
        static_cast<unsigned long long>(after_alternate.generation),
        static_cast<unsigned long long>(version_before),
        static_cast<unsigned long long>(version_after),
        static_cast<unsigned long long>(recovery_after),
        static_cast<unsigned long long>(stats.window_handoff_count),
        static_cast<unsigned long long>(stats.window_batch_count),
        static_cast<unsigned long long>(stats.window_queue_call_count),
        static_cast<unsigned long long>(stats.page_in_count),
        static_cast<unsigned long long>(stats.page_out_count),
        static_cast<unsigned>(ExactOutput(prepared)));
  }
  return valid;
}

bool RunStagedCase(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    const std::uint64_t pages, bool &unavailable) noexcept {
  return RunCase(backend, queue_counter, pages, unavailable, false);
}

} // namespace

int CheckDeviceVsmProduct(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter
        queue_counter) noexcept {
  if (!graph_multi_test::RunGraphMultiProductCases(backend, queue_counter)) {
    return 1;
  }
  bool warm_unavailable = false;
  if (!CheckDeviceVsmWarmReuse(backend, queue_counter, warm_unavailable)) {
    return 1;
  }
  if (warm_unavailable) {
    return 0;
  }
  bool binary_unavailable = false;
  if (!binary_test::RunPublicBinaryProduct(backend, binary_unavailable)) {
    return binary_unavailable ? 0 : 1;
  }
  bool capacity_unavailable = false;
  if (!binary_test::RunPublicMaximumInputProduct(backend,
                                                 capacity_unavailable)) {
    return capacity_unavailable ? 0 : 1;
  }
  if (!binary_test::RunBinaryProductCases(backend, queue_counter)) {
    return 1;
  }
  if (!binary_test::RunResidentProductCases(backend, queue_counter)) {
    return 1;
  }
  // Natural ordinary Pointwise admission uses the mapped StagedLoop owner for
  // the three small multi-epoch shapes below; forced cases remain separate.
  for (const std::uint64_t pages : {3u, 5u, 9u}) {
    bool unavailable = false;
    if (!RunStagedCase(backend, queue_counter, pages, unavailable)) {
      return 1;
    }
    if (unavailable) {
      return 0;
    }
  }
  // The shared public fixture always seals a physically short final page, so
  // its page count is odd. Use the first such count above 100,000.
  for (const std::uint64_t pages : {5u, 9u, 257u, 100001u}) {
    bool unavailable = false;
    if (!RunCase(backend, queue_counter, pages, unavailable)) {
      return 1;
    }
    if (unavailable) {
      return 0;
    }
  }
  return window_test::RunWindowProductCases(backend, queue_counter) &&
                 pointwise_dag_test::RunPointwiseDagProductCases(
                     backend, queue_counter) &&
                 graph_test::RunGraphProductCases(backend, queue_counter) &&
                 multi_scan_test::RunMultiScanProductCases(backend,
                                                           queue_counter) &&
                 scan_test::RunScanProductCases(backend, queue_counter) &&
                 scan_u32_test::RunScanProductCases(backend, queue_counter) &&
                 reduce_ops_test::RunReduceProductCases(backend,
                                                        queue_counter) &&
                 reduce_sum_test::RunReduceProductCases(backend, queue_counter)
             ? 0
             : 1;
}

} // namespace rund_node_test_device_vsm_product

#endif
