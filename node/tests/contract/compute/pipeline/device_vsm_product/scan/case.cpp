#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../persistent_product/fixture.hpp"
#include "../evidence.hpp"
#include "../route.hpp"

#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <array>
#include <cstdio>

namespace rund_node_test_device_vsm_product::scan_test {
namespace {

using rund_node_test_persistent_product::BackingRecovery;
using rund_node_test_persistent_product::BackingVersion;
using rund_node_test_persistent_product::PublicationSnapshot;
using rund_node_test_persistent_product::SnapshotPublication;

[[nodiscard]] bool RunSuccess(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    const std::uint64_t pages, const rund::kernel::ScanOp operation,
    const ScanMapCase map, bool &unavailable, std::uint64_t &retained) {
  PreparedScan prepared{};
  if (!PrepareScanProduct(backend, pages, operation, false, map, prepared,
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
  retained =
      owner == nullptr ? 0u : owner->preparation.capability.retained_bytes;
  const std::uint64_t bytes = prepared.element_count * sizeof(std::uint64_t);
  const std::uint64_t overlap = operation == rund::kernel::ScanOp::ExclusiveSum
                                    ? (pages - 1u) * sizeof(std::uint64_t)
                                    : 0u;
  const PublicationSnapshot after_primary =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot after_alternate =
      SnapshotPublication(prepared.state->alternate_pipeline);
  const rund::compute::ResidencyStats &stats =
      prepared.state->stats.pipeline.residency;
  const bool valid =
      status && queue_counter(prepared.state, queue_after) &&
      queue_after == queue_before + 1u && ExactScanOutput(prepared) &&
      owner != nullptr && owner->proof != nullptr &&
      owner->proof->topology ==
          rund::node::accel::detail::DeviceVsmTopology::Scan &&
      owner->proof->scan.semantic.op == operation &&
      owner->proof->scan.map.kind ==
          (map == ScanMapCase::None
               ? rund::node::accel::detail::DeviceVsmScanMapKind::None
           : map == ScanMapCase::Immediate
               ? rund::node::accel::detail::DeviceVsmScanMapKind::
                     AddWrapU64Immediate
               : rund::node::accel::detail::DeviceVsmScanMapKind::
                     CanonicalTotalU64) &&
      owner->proof->scan.map.immediate ==
          (map == ScanMapCase::Immediate ? TypedAdd : 0u) &&
      owner->proof->scan.stage_count ==
          1u + static_cast<std::uint32_t>(map != ScanMapCase::None) &&
      (map != ScanMapCase::Canonical ||
       owner->proof->parameter_bytes == owner->proof->plan.param_bytes) &&
      (map != ScanMapCase::Branch || (owner->proof->parameter_bytes == 0u &&
                                      owner->proof->plan.param_bytes == 0u)) &&
      owner->proof->scan.workgroup_width == 256u &&
      ExactDeviceVsmEvidence(observation, pages, pages, bytes, bytes,
                             overlap) &&
      ExactPublication(before_primary, after_primary, before_alternate,
                       after_alternate, version_before,
                       BackingVersion(*prepared.output),
                       BackingRecovery(*prepared.output)) &&
      stats.window_handoff_count == 1u && stats.window_batch_count == 1u &&
      stats.window_queue_call_count == 1u && stats.page_in_count == pages &&
      stats.page_out_count == pages && stats.backing_read_bytes == bytes &&
      stats.backing_write_bytes == bytes;
  if (!valid) {
    const bool proof_shape =
        owner != nullptr && owner->proof != nullptr &&
        owner->proof->topology ==
            rund::node::accel::detail::DeviceVsmTopology::Scan &&
        owner->proof->scan.semantic.op == operation;
    const bool proof_map =
        owner != nullptr && owner->proof != nullptr &&
        owner->proof->scan.map.kind ==
            (map == ScanMapCase::None
                 ? rund::node::accel::detail::DeviceVsmScanMapKind::None
             : map == ScanMapCase::Immediate
                 ? rund::node::accel::detail::DeviceVsmScanMapKind::
                       AddWrapU64Immediate
                 : rund::node::accel::detail::DeviceVsmScanMapKind::
                       CanonicalTotalU64) &&
        owner->proof->scan.map.immediate ==
            (map == ScanMapCase::Immediate ? TypedAdd : 0u) &&
        owner->proof->scan.stage_count ==
            1u + static_cast<std::uint32_t>(map != ScanMapCase::None);
    const bool proof_parameters =
        owner != nullptr && owner->proof != nullptr &&
        (map != ScanMapCase::Canonical ||
         owner->proof->parameter_bytes == owner->proof->plan.param_bytes) &&
        (map != ScanMapCase::Branch || (owner->proof->parameter_bytes == 0u &&
                                        owner->proof->plan.param_bytes == 0u));
    const bool exact_evidence = ExactDeviceVsmEvidence(
        observation, pages, pages, bytes, bytes, overlap);
    const bool exact_publication = ExactPublication(
        before_primary, after_primary, before_alternate, after_alternate,
        version_before, BackingVersion(*prepared.output),
        BackingRecovery(*prepared.output));
    const bool exact_stats =
        stats.window_handoff_count == 1u && stats.window_batch_count == 1u &&
        stats.window_queue_call_count == 1u && stats.page_in_count == pages &&
        stats.page_out_count == pages && stats.backing_read_bytes == bytes &&
        stats.backing_write_bytes == bytes;
    std::fprintf(
        stderr,
        "DeviceVsm Scan success backend=%u Q=%llu op=%u map=%llu status=%u "
        "route=%u/%u/%u queue=%llu/%llu topology=%u output=%u "
        "proof=%u/%u/%u evidence=%u publication=%u stats=%u "
        "param=%llu/%llu retained=%llu reason=%s\n",
        static_cast<unsigned>(backend), static_cast<unsigned long long>(pages),
        static_cast<unsigned>(operation), static_cast<unsigned long long>(map),
        static_cast<unsigned>(static_cast<bool>(status)),
        static_cast<unsigned>(observation.production_route),
        static_cast<unsigned>(observation.prepared),
        static_cast<unsigned>(observation.executed),
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_after),
        owner == nullptr || owner->proof == nullptr
            ? 0u
            : static_cast<unsigned>(owner->proof->topology),
        static_cast<unsigned>(ExactScanOutput(prepared)),
        static_cast<unsigned>(proof_shape), static_cast<unsigned>(proof_map),
        static_cast<unsigned>(proof_parameters),
        static_cast<unsigned>(exact_evidence),
        static_cast<unsigned>(exact_publication),
        static_cast<unsigned>(exact_stats),
        static_cast<unsigned long long>(owner == nullptr ||
                                                owner->proof == nullptr
                                            ? 0u
                                            : owner->proof->parameter_bytes),
        static_cast<unsigned long long>(owner == nullptr ||
                                                owner->proof == nullptr
                                            ? 0u
                                            : owner->proof->plan.param_bytes),
        static_cast<unsigned long long>(retained),
        observation.prepare_reason == nullptr ? "null"
                                              : observation.prepare_reason);
  }
  return valid;
}

[[nodiscard]] bool RunOverflow(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    bool &unavailable) {
  PreparedScan prepared{};
  if (!PrepareScanProduct(backend, 5u, rund::kernel::ScanOp::InclusiveSum, true,
                          ScanMapCase::None, prepared, unavailable)) {
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
  RouteObservation failed{};
  const rund::compute::Status status =
      RunThroughDeviceVsmProductRoute(prepared.state, failed);
  std::uint64_t queue_after = 0u;
  const auto &native = failed.evidence.native;
  const bool failure_exact =
      !status && status.reason() == rund::compute::Reason::ScanSumOverflow &&
      queue_counter(prepared.state, queue_after) &&
      queue_after == queue_before + 1u && failed.production_route &&
      failed.prepared && failed.executed && failed.evidence.final_received &&
      !failed.evidence.quarantined &&
      failed.evidence.authority_accept_count == 1u &&
      failed.evidence.pipeline_terminal_count == 2u &&
      failed.evidence.backing_publication_count == 0u &&
      native.page_count == 5u && native.generated_epochs == 5u &&
      native.completed_epochs == 2u && native.forecasted_pages == 5u &&
      native.promoted_pages == 5u && native.drained_pages == 0u &&
      native.persisted_pages == 0u && native.native_submit_count == 1u &&
      native.epoch_native_submit_count == 0u &&
      native.host_service_turn_count == 0u &&
      native.host_epoch_callback_count == 0u &&
      native.final_callback_count == 1u && !native.may_write &&
      prepared.state->stats.pipeline.residency.failed_page == 2u &&
      SnapshotPublication(prepared.state->pipeline).generation ==
          before_primary.generation &&
      SnapshotPublication(prepared.state->alternate_pipeline).generation ==
          before_alternate.generation &&
      BackingVersion(*prepared.output) == version_before &&
      BackingRecovery(*prepared.output) == 0u;
  if (!failure_exact || !SeedSafe(prepared)) {
    return false;
  }
  RouteObservation retried{};
  const rund::compute::Status retry =
      RunThroughDeviceVsmProductRoute(prepared.state, retried);
  return retry && ExactScanOutput(prepared) &&
         retried.evidence.native.native_submit_count == 1u &&
         retried.evidence.native.host_epoch_callback_count == 0u &&
         retried.evidence.backing_publication_count == 1u &&
         SnapshotPublication(prepared.state->pipeline).generation ==
             before_primary.generation + 1u &&
         SnapshotPublication(prepared.state->alternate_pipeline).generation ==
             before_alternate.generation + 1u &&
         BackingVersion(*prepared.output) == version_before + 1u;
}

} // namespace

bool RunScanProductCases(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter
        queue_counter) noexcept {
  std::array<std::uint64_t, 3u> inclusive_retained{};
  std::array<std::uint64_t, 3u> exclusive_retained{};
  std::array<std::uint64_t, 3u> mapped_inclusive_retained{};
  std::array<std::uint64_t, 3u> mapped_exclusive_retained{};
  std::array<std::uint64_t, 3u> canonical_inclusive_retained{};
  std::array<std::uint64_t, 3u> canonical_exclusive_retained{};
  std::array<std::uint64_t, 3u> branch_inclusive_retained{};
  std::array<std::uint64_t, 3u> branch_exclusive_retained{};
  std::size_t index = 0u;
  for (const std::uint64_t pages : {5u, 9u, 257u}) {
    bool unavailable = false;
    if (!RunSuccess(backend, queue_counter, pages,
                    rund::kernel::ScanOp::InclusiveSum, ScanMapCase::None,
                    unavailable, inclusive_retained[index]) ||
        !RunSuccess(backend, queue_counter, pages,
                    rund::kernel::ScanOp::ExclusiveSum, ScanMapCase::None,
                    unavailable, exclusive_retained[index]) ||
        !RunSuccess(backend, queue_counter, pages,
                    rund::kernel::ScanOp::InclusiveSum, ScanMapCase::Immediate,
                    unavailable, mapped_inclusive_retained[index]) ||
        !RunSuccess(backend, queue_counter, pages,
                    rund::kernel::ScanOp::ExclusiveSum, ScanMapCase::Immediate,
                    unavailable, mapped_exclusive_retained[index]) ||
        !RunSuccess(backend, queue_counter, pages,
                    rund::kernel::ScanOp::InclusiveSum, ScanMapCase::Canonical,
                    unavailable, canonical_inclusive_retained[index]) ||
        !RunSuccess(backend, queue_counter, pages,
                    rund::kernel::ScanOp::ExclusiveSum, ScanMapCase::Canonical,
                    unavailable, canonical_exclusive_retained[index]) ||
        !RunSuccess(backend, queue_counter, pages,
                    rund::kernel::ScanOp::InclusiveSum, ScanMapCase::Branch,
                    unavailable, branch_inclusive_retained[index]) ||
        !RunSuccess(backend, queue_counter, pages,
                    rund::kernel::ScanOp::ExclusiveSum, ScanMapCase::Branch,
                    unavailable, branch_exclusive_retained[index])) {
      return unavailable;
    }
    if (unavailable) {
      return true;
    }
    ++index;
  }
  bool unavailable = false;
  return inclusive_retained[0u] != 0u &&
         inclusive_retained[0u] == inclusive_retained[1u] &&
         inclusive_retained[0u] == inclusive_retained[2u] &&
         exclusive_retained[0u] != 0u &&
         exclusive_retained[0u] == exclusive_retained[1u] &&
         exclusive_retained[0u] == exclusive_retained[2u] &&
         mapped_inclusive_retained[0u] != 0u &&
         mapped_inclusive_retained[0u] == mapped_inclusive_retained[1u] &&
         mapped_inclusive_retained[0u] == mapped_inclusive_retained[2u] &&
         mapped_exclusive_retained[0u] != 0u &&
         mapped_exclusive_retained[0u] == mapped_exclusive_retained[1u] &&
         mapped_exclusive_retained[0u] == mapped_exclusive_retained[2u] &&
         canonical_inclusive_retained[0u] != 0u &&
         canonical_inclusive_retained[0u] == canonical_inclusive_retained[1u] &&
         canonical_inclusive_retained[0u] == canonical_inclusive_retained[2u] &&
         canonical_exclusive_retained[0u] != 0u &&
         canonical_exclusive_retained[0u] == canonical_exclusive_retained[1u] &&
         canonical_exclusive_retained[0u] == canonical_exclusive_retained[2u] &&
         branch_inclusive_retained[0u] != 0u &&
         branch_inclusive_retained[0u] == branch_inclusive_retained[1u] &&
         branch_inclusive_retained[0u] == branch_inclusive_retained[2u] &&
         branch_exclusive_retained[0u] != 0u &&
         branch_exclusive_retained[0u] == branch_exclusive_retained[1u] &&
         branch_exclusive_retained[0u] == branch_exclusive_retained[2u] &&
         RunOverflow(backend, queue_counter, unavailable);
}

} // namespace rund_node_test_device_vsm_product::scan_test

#endif
