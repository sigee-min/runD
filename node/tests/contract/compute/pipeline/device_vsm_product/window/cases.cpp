#include "fusion.hpp"
#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../persistent_product/fixture.hpp"

#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/state.hpp"

#include <array>
#include <cstdio>
#include <memory>
#include <span>
#include <vector>

namespace rund_node_test_device_vsm_product::window_test {
using namespace model;

namespace {

[[nodiscard]] bool RunWindowCase(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    const std::uint64_t pages, const Operation operation,
    const Boundary boundary, const PipelineShape shape, bool &unavailable,
    std::uint64_t &retained) {
  using namespace rund_node_test_persistent_product;
  PreparedProduct prepared{};
  if (!PrepareWindowProduct(backend, pages, operation, boundary, shape,
                            prepared, unavailable)) {
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
  const bool queue_exact = queue_counter(prepared.state, queue_after) &&
                           queue_after == queue_before + 1u;
  const PublicationSnapshot after_primary =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot after_alternate =
      SnapshotPublication(prepared.state->alternate_pipeline);
  const std::uint64_t version_after = BackingVersion(*prepared.output);
  const std::uint64_t recovery_after = BackingRecovery(*prepared.output);
  const std::uint64_t bytes = prepared.expected.size() * sizeof(std::uint32_t);
  const std::uint64_t overlap =
      (pages - 1u) * Radius * 2u * sizeof(std::uint32_t);
  const rund::compute::ResidencyStats &stats =
      prepared.state->stats.pipeline.residency;
  const auto owner = std::static_pointer_cast<
      rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner>(
      observation.owner);
  retained =
      owner == nullptr ? 0u : owner->preparation.capability.retained_bytes;
  const bool output_exact = ExactWindowOutput(prepared);
  const bool fusion_exact = owner != nullptr && owner->proof != nullptr &&
                            ExactWindowFusion(*owner->proof, shape);
  const bool valid =
      status && queue_exact && output_exact && owner != nullptr &&
      fusion_exact &&
      owner->proof->topology ==
          rund::node::accel::detail::DeviceVsmTopology::Window &&
      owner->proof->window.semantic.boundary ==
          (boundary == Boundary::Clamp ? rund::kernel::WindowBoundary::Clamp
                                       : rund::kernel::WindowBoundary::Clip) &&
      observation.evidence.native.canonical_boundary_transitions ==
          owner->proof->window.footprint.boundary_transition_count &&
      observation.evidence.native.canonical_footprint_checksum ==
          owner->proof->window.footprint.projection_checksum &&
      ExactDeviceVsmEvidence(observation, pages, pages, bytes, bytes,
                             overlap) &&
      ExactPublication(before_primary, after_primary, before_alternate,
                       after_alternate, version_before, version_after,
                       recovery_after) &&
      stats.window_handoff_count == 1u && stats.window_batch_count == 1u &&
      stats.window_queue_call_count == 1u && stats.page_in_count == pages &&
      stats.page_out_count == pages && stats.backing_read_bytes == bytes &&
      stats.backing_write_bytes == bytes && stats.page_in_bytes == bytes &&
      stats.page_out_bytes == bytes;
  if (!valid) {
    std::fprintf(
        stderr,
        "DeviceVsm Window backend=%u op=%u boundary=%u shape=%u Q=%llu "
        "status=%u "
        "route=%u/%u/%u "
        "queue=%llu/%llu overlap=%llu/%llu output=%u "
        "proof=%u path=%u stages=%u width=%u usage=%u reason=%s\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(operation),
        static_cast<unsigned>(boundary), static_cast<unsigned>(shape),
        static_cast<unsigned long long>(pages),
        static_cast<unsigned>(static_cast<bool>(status)),
        static_cast<unsigned>(observation.production_route),
        static_cast<unsigned>(observation.prepared),
        static_cast<unsigned>(observation.executed),
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_after),
        static_cast<unsigned long long>(
            observation.evidence.native.overlap_reused_bytes),
        static_cast<unsigned long long>(overlap),
        static_cast<unsigned>(output_exact),
        static_cast<unsigned>(owner != nullptr && owner->proof != nullptr),
        owner == nullptr || owner->proof == nullptr
            ? 0u
            : static_cast<unsigned>(owner->proof->window.range_path),
        owner == nullptr || owner->proof == nullptr
            ? 0u
            : static_cast<unsigned>(owner->proof->window.range_stage_count),
        owner == nullptr || owner->proof == nullptr
            ? 0u
            : static_cast<unsigned>(owner->proof->window.workgroup_width),
        owner == nullptr || owner->proof == nullptr
            ? 0u
            : static_cast<unsigned>(
                  owner->proof->residents.rows[0u].backing.usage),
        observation.prepare_reason == nullptr ? "null"
                                              : observation.prepare_reason);
  }
  return valid;
}

} // namespace

bool RunI32WindowProductCases(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    bool &unavailable) noexcept {
  unavailable = false;
  for (const bool resident : {false, true}) {
    for (const std::uint64_t pages : {2u, 3u, 5u, 4096u}) {
      I32WindowProduct prepared{};
      bool unavailable = false;
      if (!PrepareI32WindowProduct(backend, pages, resident, prepared,
                                   unavailable)) {
        return unavailable;
      }
      if (queue_counter == nullptr || prepared.output == nullptr) {
        return false;
      }
      std::uint64_t queue_before = 0u;
      if (!queue_counter(prepared.state, queue_before)) {
        return false;
      }
      const std::uint64_t version_before = I32BackingVersion(prepared.output);
      RouteObservation observation{};
      const rund::compute::Status status =
          RunThroughDeviceVsmProductRoute(prepared.state, observation, false);
      std::uint64_t queue_after = 0u;
      if (!queue_counter(prepared.state, queue_after)) {
        return false;
      }
      const std::uint64_t version_after = I32BackingVersion(prepared.output);
      const bool exact =
          ExactI32Ring(prepared, pages, status, observation, queue_before,
                       queue_after, version_before, version_after);
      if (!exact) {
        std::fprintf(
            stderr,
            "DeviceVsm I32 Window resident=%u Q=%llu status=%u "
            "queue=%llu/%llu version=%llu/%llu route=%u/%u/%u "
            "native=%llu/%llu/%llu/%llu/%llu final=%u prepare=%s\n",
            static_cast<unsigned>(resident),
            static_cast<unsigned long long>(pages),
            static_cast<unsigned>(static_cast<bool>(status)),
            static_cast<unsigned long long>(queue_before),
            static_cast<unsigned long long>(queue_after),
            static_cast<unsigned long long>(version_before),
            static_cast<unsigned long long>(version_after),
            static_cast<unsigned>(observation.production_route),
            static_cast<unsigned>(observation.prepared),
            static_cast<unsigned>(observation.executed),
            static_cast<unsigned long long>(
                observation.evidence.native.window_seed_dispatches),
            static_cast<unsigned long long>(
                observation.evidence.native.window_compute_dispatches),
            static_cast<unsigned long long>(
                observation.evidence.native.window_internal_dispatches),
            static_cast<unsigned long long>(
                prepared.state == nullptr
                    ? 0u
                    : prepared.state->stats.command_submits),
            static_cast<unsigned long long>(
                prepared.state == nullptr ? 0u
                                          : prepared.state->stats.dispatches),
            static_cast<unsigned>(observation.evidence.final_received),
            observation.prepare_reason == nullptr ? "null"
                                                  : observation.prepare_reason);
        return false;
      }
      if (pages == 4096u) {
        const auto &native = observation.evidence.native;
        const auto &stats = prepared.state->stats;
        const auto &transfers = stats.transfer_submissions;
        const std::uint64_t version_delta = version_after >= version_before
                                                ? version_after - version_before
                                                : 0u;
        std::fprintf(
            stderr,
            "DeviceVsm I32 Window resident=%u Q=%llu valid=1 "
            "native_submit=%llu command_submit=%llu generated=%llu "
            "completed=%llu epoch_submit=%llu host_turn=%llu "
            "host_callback=%llu transfer=%llu/%llu/%llu "
            "final=%llu backing_publication=%llu version_delta=%llu\n",
            static_cast<unsigned>(resident),
            static_cast<unsigned long long>(pages),
            static_cast<unsigned long long>(native.native_submit_count),
            static_cast<unsigned long long>(stats.command_submits),
            static_cast<unsigned long long>(native.generated_epochs),
            static_cast<unsigned long long>(native.completed_epochs),
            static_cast<unsigned long long>(native.epoch_native_submit_count),
            static_cast<unsigned long long>(native.host_service_turn_count),
            static_cast<unsigned long long>(native.host_epoch_callback_count),
            static_cast<unsigned long long>(transfers.host_to_device),
            static_cast<unsigned long long>(transfers.device_to_host),
            static_cast<unsigned long long>(transfers.device_to_device),
            static_cast<unsigned long long>(native.final_callback_count),
            static_cast<unsigned long long>(
                observation.evidence.backing_publication_count),
            static_cast<unsigned long long>(version_delta));
      }
    }
  }
  return true;
}

bool RunWindowProductCases(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter
        queue_counter) noexcept {
  bool i32_unavailable = false;
  if (!RunI32WindowProductCases(backend, queue_counter, i32_unavailable)) {
    return i32_unavailable;
  }
  bool ring_unavailable = false;
  if (!RunWindowRingProduct(backend, queue_counter, ring_unavailable,
                            PipelineShape::Window)) {
    return ring_unavailable;
  }
  bool fused_ring_unavailable = false;
  if (!RunWindowRingProduct(backend, queue_counter, fused_ring_unavailable,
                            PipelineShape::MapWindowMap)) {
    return fused_ring_unavailable;
  }
  for (const Operation operation :
       {Operation::Sum, Operation::Minimum, Operation::Maximum}) {
    for (const Boundary boundary : {Boundary::Clamp, Boundary::Clip}) {
      for (const PipelineShape shape :
           {PipelineShape::Window, PipelineShape::MapWindowMap,
            PipelineShape::MapDagWindowMapDag,
            PipelineShape::MapChainWindowMapChain,
            PipelineShape::MapTripleChainWindowMapTripleChain}) {
        std::array<std::uint64_t, 3u> retained{};
        std::size_t index = 0u;
        for (const std::uint64_t pages : {5u, 9u, 257u}) {
          bool unavailable = false;
          if (!RunWindowCase(backend, queue_counter, pages, operation, boundary,
                             shape, unavailable, retained[index++])) {
            return false;
          }
          if (unavailable) {
            return true;
          }
        }
        if (retained[0u] == 0u || retained[0u] != retained[1u] ||
            retained[0u] != retained[2u]) {
          return false;
        }
      }
    }
  }
  return true;
}

} // namespace rund_node_test_device_vsm_product::window_test

#endif
