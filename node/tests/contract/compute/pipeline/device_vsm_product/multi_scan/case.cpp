#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../persistent_product/fixture.hpp"
#include "../evidence.hpp"
#include "../route.hpp"

#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <cstdio>

namespace rund_node_test_device_vsm_product::multi_scan_test {
namespace {

using rund_node_test_persistent_product::BackingRecovery;
using rund_node_test_persistent_product::BackingVersion;
using rund_node_test_persistent_product::PublicationSnapshot;
using rund_node_test_persistent_product::SnapshotPublication;

[[nodiscard]] bool RunMultiScan(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    const std::uint64_t pages, const rund::kernel::ScanOp operation,
    bool &unavailable) {
  PreparedMultiScan prepared{};
  if (!PrepareMultiScanProduct(backend, pages, operation, prepared,
                               unavailable)) {
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
  RouteObservation observation{};
  const rund::compute::Status status =
      RunThroughDeviceVsmProductRoute(prepared.state, observation);
  std::uint64_t queue_after = 0u;
  const auto owner = std::static_pointer_cast<
      rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner>(
      observation.owner);
  const std::uint64_t bytes = prepared.element_count * sizeof(std::uint64_t);
  const std::uint64_t overlap = operation == rund::kernel::ScanOp::ExclusiveSum
                                    ? (pages - 1u) * sizeof(std::uint64_t)
                                    : 0u;
  const rund::compute::ResidencyStats &stats =
      prepared.state->stats.pipeline.residency;
  const bool valid =
      status && queue_counter(prepared.state, queue_after) &&
      queue_after == queue_before + 1u && ExactMultiScanOutput(prepared) &&
      owner != nullptr && owner->proof != nullptr && owner->input_count == 2u &&
      owner->proof->topology ==
          rund::node::accel::detail::DeviceVsmTopology::Scan &&
      owner->proof->scan.semantic.op == operation &&
      owner->proof->scan.map.kind ==
          rund::node::accel::detail::DeviceVsmScanMapKind::CanonicalTotalU64 &&
      owner->proof->scan.stage_count == 2u &&
      owner->proof->residents.input_count == 2u &&
      owner->proof->residents.output_count == 1u &&
      owner->proof->plan.input_buffer_count == 2u &&
      owner->proof->plan.output_buffer_count == 1u &&
      owner->proof->fixed_common_storage &&
      ExactDeviceVsmEvidence(observation, pages, pages, 2u * bytes, bytes,
                             overlap) &&
      ExactPublication(before_primary,
                       SnapshotPublication(prepared.state->pipeline),
                       before_alternate,
                       SnapshotPublication(prepared.state->alternate_pipeline),
                       version_before, BackingVersion(*prepared.output),
                       BackingRecovery(*prepared.output)) &&
      stats.window_handoff_count == 1u && stats.window_batch_count == 1u &&
      stats.window_queue_call_count == 1u &&
      stats.page_in_count == 2u * pages && stats.page_out_count == pages &&
      stats.backing_read_bytes == 2u * bytes &&
      stats.backing_write_bytes == bytes;
  if (!valid) {
    std::fprintf(
        stderr,
        "DeviceVsm multi Scan backend=%u Q=%llu op=%u valid=%u "
        "queue=%llu/%llu submit=%llu epoch_submit=%llu host_turn=%llu "
        "host_callback=%llu final=%u reason=%s\n",
        static_cast<unsigned>(backend), static_cast<unsigned long long>(pages),
        static_cast<unsigned>(operation), static_cast<unsigned>(valid),
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_after),
        static_cast<unsigned long long>(
            observation.evidence.native.native_submit_count),
        static_cast<unsigned long long>(
            observation.evidence.native.epoch_native_submit_count),
        static_cast<unsigned long long>(
            observation.evidence.native.host_service_turn_count),
        static_cast<unsigned long long>(
            observation.evidence.native.host_epoch_callback_count),
        static_cast<unsigned>(observation.evidence.final_received),
        observation.prepare_reason == nullptr ? "null"
                                              : observation.prepare_reason);
  }
  return valid;
}

} // namespace

bool RunMultiScanProductCases(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter
        queue_counter) noexcept {
  bool rejection_unavailable = false;
  if (!RejectWideMultiScan(backend, rejection_unavailable) ||
      rejection_unavailable) {
    return rejection_unavailable;
  }
  for (const rund::kernel::ScanOp operation :
       {rund::kernel::ScanOp::InclusiveSum,
        rund::kernel::ScanOp::ExclusiveSum}) {
    bool unavailable = false;
    if (!RunMaximumInputScan(backend, queue_counter, operation, unavailable)) {
      return unavailable;
    }
    if (unavailable) {
      return true;
    }
  }
  for (const std::uint64_t pages : {5u, 9u, 257u}) {
    for (const rund::kernel::ScanOp operation :
         {rund::kernel::ScanOp::InclusiveSum,
          rund::kernel::ScanOp::ExclusiveSum}) {
      bool unavailable = false;
      if (!RunMultiScan(backend, queue_counter, pages, operation,
                        unavailable)) {
        return unavailable;
      }
      if (unavailable) {
        return true;
      }
    }
  }
  return true;
}

} // namespace rund_node_test_device_vsm_product::multi_scan_test

#endif
