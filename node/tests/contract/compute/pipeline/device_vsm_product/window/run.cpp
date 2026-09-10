#include "fusion.hpp"
#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../persistent_product/fixture.hpp"

#include "src/compute/device/residency/pool.hpp"
#include "src/compute/device/state.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/state.hpp"

#include "src/accel/kernel/fault.hpp"
#include "src/hash/fnv.hpp"

#include <array>
#include <cstdio>
#include <memory>
#include <span>

namespace rund_node_test_device_vsm_product::window_test {

using rund::compute::ResidencyStats;
using rund_node_test_persistent_product::BackingRecovery;
using rund_node_test_persistent_product::BackingVersion;
using rund_node_test_persistent_product::NativeQueueCounter;
using rund_node_test_persistent_product::PreparedProduct;
using rund_node_test_persistent_product::PublicationSnapshot;
using rund_node_test_persistent_product::SnapshotPublication;
using DeviceVsmProductOwner =
    rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;
using WindowRingPlan = rund::node::accel::detail::DeviceVsmWindowRingPlan;
using namespace model;

[[nodiscard]] bool RunWindowRingProduct(const rund::compute::Backend backend,
                                        const NativeQueueCounter queue_counter,
                                        bool &unavailable,
                                        const PipelineShape shape) {
  using namespace rund_node_test_persistent_product;
  PreparedProduct prepared{};
  if (!PrepareRingShape(backend, shape, prepared, unavailable)) {
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
  RouteObservation cold{};
  const rund::compute::Status cold_status =
      RunThroughDeviceVsmProductRoute(prepared.state, cold, false);
  const bool cold_output = cold_status && ExactWindowOutput(prepared);
  const PublicationSnapshot cold_primary =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot cold_alternate =
      SnapshotPublication(prepared.state->alternate_pipeline);
  const std::uint64_t cold_version = BackingVersion(*prepared.output);
  const std::uint64_t cold_recovery = BackingRecovery(*prepared.output);
  std::uint64_t cold_queue = 0u;
  const bool cold_queue_ok = queue_counter(prepared.state, cold_queue);
  const rund::compute::Stats cold_stats = prepared.state->stats;
  const auto cold_owner =
      std::static_pointer_cast<DeviceVsmProductOwner>(cold.owner);
  WindowRingPlan plan{};
  const std::uint64_t bytes = prepared.expected.size() * sizeof(std::uint32_t);
  const std::uint64_t overlap =
      (WindowRingPages - 1u) * Radius * 2u * sizeof(std::uint32_t);
  const std::uint64_t expected_hash =
      rund::node::hash_detail::HashBytes(prepared.expected.data(), bytes);
  const bool cold_proof = ExactRingProof(prepared, cold, cold_owner, plan);
  const bool cold_fusion = cold_owner != nullptr &&
                           cold_owner->proof != nullptr &&
                           ExactWindowFusion(*cold_owner->proof, shape);
  const bool cold_status_ok = static_cast<bool>(cold_status);
  const bool cold_queue_next = cold_queue == queue_before + 1u;
  const bool cold_stats_ok =
      ExactRingStats(cold_stats, WindowRingPages, bytes, 1u);
  const bool cold_hash_ok =
      cold_stats.output_hash == expected_hash && cold_stats.output_hash != 0u;
  const bool cold_native_ok = ExactWindowRingEvidence(
      cold, WindowRingPages, bytes, bytes, overlap, plan);
  const bool cold_counts_ok = cold.evidence.cold_prepare_count == 1u &&
                              cold.evidence.warm_rearm_count == 0u;
  const bool cold_publication_ok = ExactPublication(
      before_primary, cold_primary, before_alternate, cold_alternate,
      version_before, cold_version, cold_recovery);
  const bool cold_exact = cold_status_ok && cold_output && cold_queue_ok &&
                          cold_queue_next && cold_stats_ok && cold_hash_ok &&
                          cold_proof && cold_fusion && cold_native_ok &&
                          cold_counts_ok && cold_publication_ok;
  if (!cold_exact) {
    const auto &cold_evidence = cold.evidence;
    const auto &cold_native = cold_evidence.native;
    const auto &cold_residency = cold_stats.pipeline.residency;
    const std::uint64_t expected_epochs =
        WindowRingPages / 2u +
        static_cast<std::uint64_t>(WindowRingPages % 2u != 0u);
    const auto *cold_proof_value =
        cold_owner == nullptr ? nullptr : cold_owner->proof.get();
    const bool proof_ring_equal =
        cold_proof_value != nullptr && cold_proof_value->window.ring == plan;
    std::fprintf(
        stderr,
        "DeviceVsm WindowRing cold shape=%u checks status=%u output=%u "
        "queue=%u/%u stats=%u hash=%u proof=%u fusion=%u native=%u "
        "counts=%u publication=%u\n",
        static_cast<unsigned>(shape), static_cast<unsigned>(cold_status_ok),
        static_cast<unsigned>(cold_output),
        static_cast<unsigned>(cold_queue_ok),
        static_cast<unsigned>(cold_queue_next),
        static_cast<unsigned>(cold_stats_ok),
        static_cast<unsigned>(cold_hash_ok), static_cast<unsigned>(cold_proof),
        static_cast<unsigned>(cold_fusion),
        static_cast<unsigned>(cold_native_ok),
        static_cast<unsigned>(cold_counts_ok),
        static_cast<unsigned>(cold_publication_ok));
    std::fprintf(
        stderr,
        "DeviceVsm WindowRing cold stats cmd=%llu/%u dispatch=%llu/%u "
        "final_dispatch=%llu/%u external=%llu/0 logical=%llu/%llu "
        "pages=%llu/%u frame=%llu/2 epochs=%llu/%llu handoff=%llu/1 "
        "batch=%llu/1 queue_call=%llu/1 page_in=%llu/%u page_out=%llu/%u\n",
        static_cast<unsigned long long>(cold_stats.command_submits), 1u,
        static_cast<unsigned long long>(cold_stats.dispatches), 1u,
        static_cast<unsigned long long>(cold_stats.final_dispatches), 1u,
        static_cast<unsigned long long>(cold_stats.external_roundtrip_bytes),
        static_cast<unsigned long long>(cold_residency.logical_bytes),
        static_cast<unsigned long long>(RingLogicalBytes(bytes)),
        static_cast<unsigned long long>(cold_residency.page_count),
        static_cast<unsigned>(WindowRingPages),
        static_cast<unsigned long long>(cold_residency.frame_capacity),
        static_cast<unsigned long long>(cold_residency.epoch_count),
        static_cast<unsigned long long>(expected_epochs),
        static_cast<unsigned long long>(cold_residency.window_handoff_count),
        static_cast<unsigned long long>(cold_residency.window_batch_count),
        static_cast<unsigned long long>(cold_residency.window_queue_call_count),
        static_cast<unsigned long long>(cold_residency.page_in_count),
        static_cast<unsigned>(WindowRingPages),
        static_cast<unsigned long long>(cold_residency.page_out_count),
        static_cast<unsigned>(WindowRingPages));
    std::fprintf(
        stderr,
        "DeviceVsm WindowRing cold bytes read=%llu/%llu write=%llu/%llu "
        "page_in=%llu/%llu page_out=%llu/%llu peak=%llu/%llu\n",
        static_cast<unsigned long long>(cold_residency.backing_read_bytes),
        static_cast<unsigned long long>(bytes),
        static_cast<unsigned long long>(cold_residency.backing_write_bytes),
        static_cast<unsigned long long>(bytes),
        static_cast<unsigned long long>(cold_residency.page_in_bytes),
        static_cast<unsigned long long>(bytes),
        static_cast<unsigned long long>(cold_residency.page_out_bytes),
        static_cast<unsigned long long>(bytes),
        static_cast<unsigned long long>(cold_residency.resident_frames_peak),
        static_cast<unsigned long long>(RingPeak(WindowRingPages)));
    std::fprintf(
        stderr,
        "DeviceVsm WindowRing cold proof=%u owner=%u topology=%u ring=%u "
        "width=%u ring_equal=%u physical=%u native_submit=%u "
        "plan page=%u frame=%u payload=%u halo=%u checksum=%u config=%u "
        "state=%llu scratch=%llu\n",
        static_cast<unsigned>(cold_proof),
        static_cast<unsigned>(cold_owner != nullptr),
        cold_proof_value == nullptr
            ? 0u
            : static_cast<unsigned>(cold_proof_value->topology),
        cold_proof_value == nullptr
            ? 0u
            : static_cast<unsigned>(cold_proof_value->window.ring.gpu_owned),
        cold_proof_value == nullptr ? 0u : cold_proof_value->width,
        static_cast<unsigned>(proof_ring_equal),
        static_cast<unsigned>(
            cold_owner != nullptr &&
            cold_owner->preparation.capability.physical_ring_storage),
        static_cast<unsigned>(
            cold_owner != nullptr &&
            cold_owner->preparation.capability.one_native_submit),
        plan.page_count, plan.frame_elements, plan.payload_elements,
        plan.halo_elements, plan.schedule_checksum, plan.config_stride,
        static_cast<unsigned long long>(plan.state_bytes),
        static_cast<unsigned long long>(plan.scratch_bytes));
    std::fprintf(
        stderr,
        "DeviceVsm WindowRing cold publication=%u primary=%llu/%llu->%llu/%llu "
        "alternate=%llu/%llu->%llu/%llu version=%llu/%llu recovery=%llu\n",
        static_cast<unsigned>(cold_publication_ok),
        static_cast<unsigned long long>(before_primary.generation),
        static_cast<unsigned long long>(before_primary.payload_epoch),
        static_cast<unsigned long long>(cold_primary.generation),
        static_cast<unsigned long long>(cold_primary.payload_epoch),
        static_cast<unsigned long long>(before_alternate.generation),
        static_cast<unsigned long long>(before_alternate.payload_epoch),
        static_cast<unsigned long long>(cold_alternate.generation),
        static_cast<unsigned long long>(cold_alternate.payload_epoch),
        static_cast<unsigned long long>(version_before),
        static_cast<unsigned long long>(cold_version),
        static_cast<unsigned long long>(cold_recovery));
    std::fprintf(
        stderr,
        "DeviceVsm WindowRing cold backend=%u status=%u exec=%u reason=%u "
        "poison=%u route=%u/%u/%u queue=%llu/%llu output=%u "
        "hash=%llu/%llu final=%u quarantine=%u authority=%llu "
        "pipeline=%llu backing=%llu prepare=%s\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned>(static_cast<bool>(cold_status)),
        static_cast<unsigned>(cold.execution_status), cold.execution_reason,
        static_cast<unsigned>(cold.execution_poison),
        static_cast<unsigned>(cold.production_route),
        static_cast<unsigned>(cold.prepared),
        static_cast<unsigned>(cold.executed),
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(cold_queue),
        static_cast<unsigned>(cold_output),
        static_cast<unsigned long long>(cold_stats.output_hash),
        static_cast<unsigned long long>(expected_hash),
        static_cast<unsigned>(cold_evidence.final_received),
        static_cast<unsigned>(cold_evidence.quarantined),
        static_cast<unsigned long long>(cold_evidence.authority_accept_count),
        static_cast<unsigned long long>(cold_evidence.pipeline_terminal_count),
        static_cast<unsigned long long>(
            cold_evidence.backing_publication_count),
        cold.prepare_reason == nullptr ? "null" : cold.prepare_reason);
    std::fprintf(
        stderr,
        "DeviceVsm WindowRing cold native page=%llu generated=%llu "
        "completed=%llu forecast=%llu promote=%llu drain=%llu persist=%llu "
        "submit=%llu epoch_submit=%llu payload=%llu host_turn=%llu "
        "host_callback=%llu final_callback=%llu seed=%llu compute=%llu "
        "internal=%llu reuse=%llu checksum=%llu roundtrips=%llu max_live=%u "
        "may_write=%u\n",
        static_cast<unsigned long long>(cold_native.page_count),
        static_cast<unsigned long long>(cold_native.generated_epochs),
        static_cast<unsigned long long>(cold_native.completed_epochs),
        static_cast<unsigned long long>(cold_native.forecasted_pages),
        static_cast<unsigned long long>(cold_native.promoted_pages),
        static_cast<unsigned long long>(cold_native.drained_pages),
        static_cast<unsigned long long>(cold_native.persisted_pages),
        static_cast<unsigned long long>(cold_native.native_submit_count),
        static_cast<unsigned long long>(cold_native.epoch_native_submit_count),
        static_cast<unsigned long long>(cold_native.payload_dispatch_count),
        static_cast<unsigned long long>(cold_native.host_service_turn_count),
        static_cast<unsigned long long>(cold_native.host_epoch_callback_count),
        static_cast<unsigned long long>(cold_native.final_callback_count),
        static_cast<unsigned long long>(cold_native.window_seed_dispatches),
        static_cast<unsigned long long>(cold_native.window_compute_dispatches),
        static_cast<unsigned long long>(cold_native.window_internal_dispatches),
        static_cast<unsigned long long>(cold_native.ring_reuse_transitions),
        static_cast<unsigned long long>(cold_native.ring_schedule_checksum),
        static_cast<unsigned long long>(cold_native.ring_round_trips),
        static_cast<unsigned>(cold_native.max_live_frames),
        static_cast<unsigned>(cold_native.may_write));
    return false;
  }

  RouteObservation warm{};
  const rund::compute::Status warm_status =
      RunThroughDeviceVsmProductRoute(prepared.state, warm, false);
  const bool warm_output = warm_status && ExactWindowOutput(prepared);
  const PublicationSnapshot warm_primary =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot warm_alternate =
      SnapshotPublication(prepared.state->alternate_pipeline);
  const std::uint64_t warm_version = BackingVersion(*prepared.output);
  const std::uint64_t warm_recovery = BackingRecovery(*prepared.output);
  std::uint64_t warm_queue = 0u;
  const bool warm_queue_ok = queue_counter(prepared.state, warm_queue);
  const rund::compute::Stats warm_stats = prepared.state->stats;
  const auto warm_owner =
      std::static_pointer_cast<DeviceVsmProductOwner>(warm.owner);
  WindowRingPlan warm_plan{};
  const bool warm_fusion = warm_owner != nullptr &&
                           warm_owner->proof != nullptr &&
                           ExactWindowFusion(*warm_owner->proof, shape);
  const bool warm_exact =
      warm_status && warm_output && warm_queue_ok &&
      warm_queue == cold_queue + 1u && cold.owner.get() == warm.owner.get() &&
      cold_owner == warm_owner &&
      ExactRingProof(prepared, warm, warm_owner, warm_plan) && warm_fusion &&
      plan == warm_plan &&
      ExactRingStats(warm_stats, WindowRingPages, bytes, 1u) &&
      warm_stats.output_hash == expected_hash &&
      ExactWindowRingEvidence(warm, WindowRingPages, bytes, bytes, overlap,
                              warm_plan) &&
      warm.evidence.cold_prepare_count == 1u &&
      warm.evidence.warm_rearm_count == 1u &&
      ExactPublication(cold_primary, warm_primary, cold_alternate,
                       warm_alternate, cold_version, warm_version,
                       warm_recovery);
  if (!warm_exact) {
    std::fprintf(
        stderr,
        "DeviceVsm WindowRing warm backend=%u status=%u queue=%llu/%llu "
        "output=%u hash=%llu/%llu cold/rearm=%llu/%llu reason=%s\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned>(static_cast<bool>(warm_status)),
        static_cast<unsigned long long>(cold_queue),
        static_cast<unsigned long long>(warm_queue),
        static_cast<unsigned>(warm_output),
        static_cast<unsigned long long>(warm_stats.output_hash),
        static_cast<unsigned long long>(expected_hash),
        static_cast<unsigned long long>(warm.evidence.cold_prepare_count),
        static_cast<unsigned long long>(warm.evidence.warm_rearm_count),
        warm.prepare_reason == nullptr ? "null" : warm.prepare_reason);
    return false;
  }

  PreparedProduct failed_product{};
  bool loss_unavailable = false;
  if (!PrepareRingShape(backend, shape, failed_product, loss_unavailable)) {
    return loss_unavailable;
  }
  const PublicationSnapshot loss_before_primary =
      SnapshotPublication(failed_product.state->pipeline);
  const PublicationSnapshot loss_before_alternate =
      SnapshotPublication(failed_product.state->alternate_pipeline);
  const std::uint64_t loss_version = BackingVersion(*failed_product.output);
  const std::uint64_t loss_bytes =
      failed_product.expected.size() * sizeof(std::uint32_t);
  std::uint64_t loss_queue_before = 0u;
  if (!queue_counter(failed_product.state, loss_queue_before) ||
      failed_product.state->pipeline == nullptr ||
      failed_product.state->pipeline->device == nullptr) {
    return false;
  }
  auto *const native = rund::compute::detail::accel_device(
      *failed_product.state->pipeline->device);
  if (native == nullptr ||
      !rund::node::accel::detail::InjectNativeDeviceLostOnce(native->pick)) {
    return false;
  }
  RouteObservation loss{};
  const rund::compute::Status loss_status =
      RunThroughDeviceVsmProductRoute(failed_product.state, loss, false);
  std::uint64_t loss_queue_after = 0u;
  const bool loss_queue_ok =
      queue_counter(failed_product.state, loss_queue_after);
  const std::uint64_t loss_version_after =
      BackingVersion(*failed_product.output);
  const std::uint64_t loss_recovery = BackingRecovery(*failed_product.output);
  const rund::compute::Stats loss_stats = failed_product.state->stats;
  const auto loss_owner =
      std::static_pointer_cast<DeviceVsmProductOwner>(loss.owner);
  const bool loss_fusion = loss_owner != nullptr &&
                           loss_owner->proof != nullptr &&
                           ExactWindowFusion(*loss_owner->proof, shape);
  const bool loss_exact =
      ExactRingFailure(loss_status, loss, loss_bytes, loss_version,
                       loss_version_after, loss_recovery, loss_queue_before,
                       loss_queue_after, loss_stats) &&
      loss_fusion && loss_queue_ok &&
      SnapshotPublication(failed_product.state->pipeline).generation ==
          loss_before_primary.generation &&
      SnapshotPublication(failed_product.state->pipeline).payload_epoch ==
          loss_before_primary.payload_epoch &&
      SnapshotPublication(failed_product.state->alternate_pipeline)
              .generation == loss_before_alternate.generation &&
      SnapshotPublication(failed_product.state->alternate_pipeline)
              .payload_epoch == loss_before_alternate.payload_epoch;
  if (!loss_exact) {
    std::fprintf(stderr,
                 "DeviceVsm WindowRing loss backend=%u status=%u reason=%u "
                 "queue=%llu/%llu quarantine=%u recovery=%llu/%llu\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(static_cast<bool>(loss_status)),
                 loss.execution_reason,
                 static_cast<unsigned long long>(loss_queue_before),
                 static_cast<unsigned long long>(loss_queue_after),
                 static_cast<unsigned>(loss.evidence.quarantined),
                 static_cast<unsigned long long>(loss_recovery),
                 static_cast<unsigned long long>(loss_bytes));
  }
  return loss_exact;
}

} // namespace rund_node_test_device_vsm_product::window_test

#endif
