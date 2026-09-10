#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../persistent_product/fixture.hpp"
#include "../evidence.hpp"
#include "../route.hpp"

#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <array>
#include <cstdio>

namespace rund_node_test_device_vsm_product::binary_test {
namespace {

using rund_node_test_persistent_product::BackingRecovery;
using rund_node_test_persistent_product::BackingVersion;
using rund_node_test_persistent_product::PublicationSnapshot;
using rund_node_test_persistent_product::SnapshotPublication;

[[nodiscard]] bool RunBinary(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    const std::uint64_t pages, bool &unavailable) {
  PreparedBinary prepared{};
  if (!PrepareBinaryProduct(backend, pages, prepared, unavailable)) {
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
  const std::uint64_t bytes = prepared.element_count * sizeof(std::uint32_t);
  const PublicationSnapshot after_primary =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot after_alternate =
      SnapshotPublication(prepared.state->alternate_pipeline);
  const rund::compute::ResidencyStats &stats =
      prepared.state->stats.pipeline.residency;
  const bool valid =
      status && queue_counter(prepared.state, queue_after) &&
      queue_after == queue_before + 1u && ExactBinaryOutput(prepared) &&
      owner != nullptr && owner->proof != nullptr && owner->input_count == 2u &&
      owner->proof->residents.input_count == 2u &&
      owner->proof->residents.output_count == 1u &&
      owner->proof->residents.count == 3u &&
      owner->proof->plan.input_buffer_count == 2u &&
      owner->proof->plan.output_buffer_count == 1u &&
      owner->proof->fixed_common_storage && owner->evidence != nullptr &&
      owner->evidence->public_resident_input_count == 0u &&
      owner->evidence->whole_run_staged_input_count == 2u &&
      !owner->evidence->public_resident_output &&
      owner->evidence->whole_run_staged_output &&
      !owner->evidence->bounded_external_page_service &&
      ExactDeviceVsmEvidence(observation, pages, pages, 2u * bytes, bytes,
                             0u) &&
      ExactPublication(before_primary, after_primary, before_alternate,
                       after_alternate, version_before,
                       BackingVersion(*prepared.output),
                       BackingRecovery(*prepared.output)) &&
      stats.window_handoff_count == 1u && stats.window_batch_count == 1u &&
      stats.window_queue_call_count == 1u &&
      stats.page_in_count == 2u * pages && stats.page_out_count == pages &&
      stats.backing_read_bytes == 2u * bytes &&
      stats.backing_write_bytes == bytes && stats.page_in_bytes == 2u * bytes &&
      stats.page_out_bytes == bytes;
  std::fprintf(
      stderr,
      "DeviceVsm binary product backend=%u Q=%llu valid=%u "
      "queue=%llu/%llu submit=%llu epoch_submit=%llu host_turn=%llu "
      "host_callback=%llu generated=%llu completed=%llu final=%u "
      "public=%u/%u staged=%u/%u external=%u\n",
      static_cast<unsigned>(backend), static_cast<unsigned long long>(pages),
      static_cast<unsigned>(valid),
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
      static_cast<unsigned long long>(
          observation.evidence.native.generated_epochs),
      static_cast<unsigned long long>(
          observation.evidence.native.completed_epochs),
      static_cast<unsigned>(observation.evidence.final_received),
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
  return valid;
}

} // namespace

bool RunBinaryProductCases(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter
        queue_counter) noexcept {
  for (const std::uint64_t pages : {5u, 9u, 257u}) {
    bool unavailable = false;
    if (!RunBinary(backend, queue_counter, pages, unavailable)) {
      return unavailable;
    }
    if (unavailable) {
      return true;
    }
  }
  return true;
}

} // namespace rund_node_test_device_vsm_product::binary_test

#endif
