#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../persistent_product/fixture.hpp"
#include "../evidence.hpp"
#include "../route.hpp"

#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <algorithm>
#include <array>
#include <cstdio>

namespace rund_node_test_device_vsm_product::pointwise_dag_test {
namespace {

using rund_node_test_persistent_product::BackingRecovery;
using rund_node_test_persistent_product::BackingVersion;
using rund_node_test_persistent_product::PublicationSnapshot;
using rund_node_test_persistent_product::SnapshotPublication;

[[nodiscard]] bool RunSuccess(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    const std::uint64_t pages, const PointwiseDagType type, bool &unavailable,
    std::uint64_t &retained) {
  PreparedPointwiseDag prepared{};
  if (!PreparePointwiseDagProduct(backend, pages, type, prepared,
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
  const std::uint64_t bytes = prepared.element_count * prepared.element_bytes;
  const PublicationSnapshot after_primary =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot after_alternate =
      SnapshotPublication(prepared.state->alternate_pipeline);
  const rund::compute::ResidencyStats &stats =
      prepared.state->stats.pipeline.residency;
  const bool valid =
      status && queue_counter(prepared.state, queue_after) &&
      queue_after == queue_before + 1u && ExactPointwiseDagOutput(prepared) &&
      owner != nullptr && owner->proof != nullptr &&
      owner->proof->topology ==
          rund::node::accel::detail::DeviceVsmTopology::Pointwise &&
      owner->proof->fixed_common_storage &&
      owner->proof->parameter_bytes == 0u &&
      owner->proof->plan.param_bytes == 0u &&
      owner->proof->plan.input_buffer_count == 1u &&
      owner->proof->plan.output_buffer_count == 1u &&
      ExactDeviceVsmEvidence(observation, pages, pages, bytes, bytes, 0u) &&
      ExactPublication(before_primary, after_primary, before_alternate,
                       after_alternate, version_before,
                       BackingVersion(*prepared.output),
                       BackingRecovery(*prepared.output)) &&
      stats.window_handoff_count == 1u && stats.window_batch_count == 1u &&
      stats.window_queue_call_count == 1u && stats.page_in_count == pages &&
      stats.page_out_count == pages && stats.backing_read_bytes == bytes &&
      stats.backing_write_bytes == bytes;
  if (valid) {
    std::fprintf(stderr,
                 "DeviceVsm pointwise DAG backend=%u type=%u Q=%llu "
                 "submit=%llu "
                 "epoch_submit=%llu host_turn=%llu host_callback=%llu final=%u "
                 "retained=%llu\n",
                 static_cast<unsigned>(backend), static_cast<unsigned>(type),
                 static_cast<unsigned long long>(pages),
                 static_cast<unsigned long long>(
                     observation.evidence.native.native_submit_count),
                 static_cast<unsigned long long>(
                     observation.evidence.native.epoch_native_submit_count),
                 static_cast<unsigned long long>(
                     observation.evidence.native.host_service_turn_count),
                 static_cast<unsigned long long>(
                     observation.evidence.native.host_epoch_callback_count),
                 static_cast<unsigned>(observation.evidence.final_received),
                 static_cast<unsigned long long>(retained));
  }
  if (!valid) {
    std::fprintf(
        stderr,
        "DeviceVsm pointwise DAG backend=%u type=%u Q=%llu status=%u "
        "route=%u/%u/%u queue=%llu/%llu topology=%u output=%u retained=%llu "
        "reason=%s\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(type),
        static_cast<unsigned long long>(pages),
        static_cast<unsigned>(static_cast<bool>(status)),
        static_cast<unsigned>(observation.production_route),
        static_cast<unsigned>(observation.prepared),
        static_cast<unsigned>(observation.executed),
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_after),
        owner == nullptr || owner->proof == nullptr
            ? 0u
            : static_cast<unsigned>(owner->proof->topology),
        static_cast<unsigned>(ExactPointwiseDagOutput(prepared)),
        static_cast<unsigned long long>(retained),
        observation.prepare_reason == nullptr ? "null"
                                              : observation.prepare_reason);
  }
  return valid;
}

} // namespace

bool RunPointwiseDagProductCases(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter
        queue_counter) noexcept {
  std::array<std::array<std::uint64_t, 3u>, 2u> retained{};
  for (const PointwiseDagType type :
       {PointwiseDagType::U32, PointwiseDagType::U64}) {
    std::size_t index = 0u;
    for (const std::uint64_t pages : {5u, 9u, 257u}) {
      bool unavailable = false;
      const std::size_t type_index = static_cast<std::size_t>(type);
      if (!RunSuccess(backend, queue_counter, pages, type, unavailable,
                      retained[type_index][index])) {
        return unavailable;
      }
      if (unavailable) {
        return true;
      }
      ++index;
    }
  }
  return std::all_of(retained.begin(), retained.end(), [](const auto &values) {
    return values[0u] != 0u && values[0u] == values[1u] &&
           values[0u] == values[2u];
  });
}

} // namespace rund_node_test_device_vsm_product::pointwise_dag_test

#endif
