#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../persistent_product/fixture.hpp"
#include "../evidence.hpp"
#include "../route.hpp"

#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <array>
#include <cstdio>

namespace rund_node_test_device_vsm_product::reduce_ops_test {
namespace {

using rund_node_test_persistent_product::BackingRecovery;
using rund_node_test_persistent_product::BackingVersion;
using rund_node_test_persistent_product::PublicationSnapshot;
using rund_node_test_persistent_product::SnapshotPublication;

[[nodiscard]] bool RunSuccess(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    const std::uint64_t pages, const std::uint32_t element_bytes,
    const rund::kernel::ReduceOp operation, bool &unavailable,
    std::uint64_t &retained) {
  PreparedReduce prepared{};
  if (!PrepareReduceProduct(backend, pages, element_bytes, operation, prepared,
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
  const std::uint64_t input_bytes = prepared.element_count * element_bytes;
  const PublicationSnapshot after_primary =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot after_alternate =
      SnapshotPublication(prepared.state->alternate_pipeline);
  const rund::compute::ResidencyStats &stats =
      prepared.state->stats.pipeline.residency;
  const bool valid =
      status && queue_counter(prepared.state, queue_after) &&
      queue_after == queue_before + 1u && ExactReduceOutput(prepared) &&
      owner != nullptr && owner->proof != nullptr &&
      owner->proof->topology ==
          rund::node::accel::detail::DeviceVsmTopology::Reduce &&
      owner->proof->reduce.semantic.op == operation &&
      owner->proof->reduce.semantic.element_bytes == element_bytes &&
      owner->proof->reduce.workgroup_width == 256u &&
      ExactDeviceVsmEvidence(observation, pages, 1u, input_bytes, element_bytes,
                             0u) &&
      ExactPublication(before_primary, after_primary, before_alternate,
                       after_alternate, version_before,
                       BackingVersion(*prepared.output),
                       BackingRecovery(*prepared.output)) &&
      stats.window_handoff_count == 1u && stats.window_batch_count == 1u &&
      stats.window_queue_call_count == 1u && stats.page_in_count == pages &&
      stats.page_out_count == 1u && stats.backing_read_bytes == input_bytes &&
      stats.backing_write_bytes == element_bytes;
  if (!valid) {
    std::fprintf(
        stderr,
        "DeviceVsm Reduce op backend=%u op=%u Q=%llu bytes=%u status=%u "
        "route=%u/%u/%u execute=%u/%u/%u queue=%llu/%llu output=%u "
        "native=%llu/%llu/%llu/%llu may_write=%u quarantine=%u reason=%s\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(operation),
        static_cast<unsigned long long>(pages), element_bytes,
        static_cast<unsigned>(static_cast<bool>(status)),
        static_cast<unsigned>(observation.production_route),
        static_cast<unsigned>(observation.prepared),
        static_cast<unsigned>(observation.executed),
        static_cast<unsigned>(observation.execution_status),
        observation.execution_reason,
        static_cast<unsigned>(observation.execution_poison),
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_after),
        static_cast<unsigned>(ExactReduceOutput(prepared)),
        static_cast<unsigned long long>(
            observation.evidence.native.native_submit_count),
        static_cast<unsigned long long>(
            observation.evidence.native.generated_epochs),
        static_cast<unsigned long long>(
            observation.evidence.native.completed_epochs),
        static_cast<unsigned long long>(
            observation.evidence.native.final_callback_count),
        static_cast<unsigned>(observation.evidence.native.may_write),
        static_cast<unsigned>(observation.evidence.quarantined),
        observation.prepare_reason == nullptr ? "null"
                                              : observation.prepare_reason);
  }
  return valid;
}

[[nodiscard]] bool RunOperation(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    const std::uint32_t element_bytes, const rund::kernel::ReduceOp operation) {
  std::array<std::uint64_t, 3u> retained{};
  std::size_t index = 0u;
  for (const std::uint64_t pages : {5u, 9u, 257u}) {
    bool unavailable = false;
    if (!RunSuccess(backend, queue_counter, pages, element_bytes, operation,
                    unavailable, retained[index])) {
      return unavailable;
    }
    if (unavailable) {
      return true;
    }
    ++index;
  }
  return retained[0u] != 0u && retained[0u] == retained[1u] &&
         retained[0u] == retained[2u];
}

} // namespace

bool RunReduceProductCases(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter
        queue_counter) noexcept {
  for (const std::uint32_t element_bytes :
       {static_cast<std::uint32_t>(sizeof(std::uint32_t)),
        static_cast<std::uint32_t>(sizeof(std::uint64_t))}) {
    for (const rund::kernel::ReduceOp operation :
         {rund::kernel::ReduceOp::CountNonzero, rund::kernel::ReduceOp::Min,
          rund::kernel::ReduceOp::Max}) {
      if (!RunOperation(backend, queue_counter, element_bytes, operation)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace rund_node_test_device_vsm_product::reduce_ops_test

#endif
