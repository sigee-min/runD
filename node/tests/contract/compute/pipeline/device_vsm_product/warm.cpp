#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "evidence.hpp"
#include "route.hpp"

#include "../persistent_product/fixture.hpp"

#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/local.hpp"
#include "src/compute/virtual/state.hpp"

#include <cstdio>

namespace rund_node_test_device_vsm_product {

bool CheckDeviceVsmWarmReuse(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    bool &unavailable) noexcept {
  using namespace rund_node_test_persistent_product;
  constexpr std::uint64_t pages = 9u;
  PreparedProduct prepared{};
  if (!PrepareProduct(backend, (pages + 1u) / 2u, prepared, unavailable)) {
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
  const rund::compute::Status conditioned =
      RunThroughDeviceVsmProductRoute(prepared.state, cold);
  const rund::compute::Status samples =
      conditioned ? rund::compute::detail::begin_virtual_pipeline_samples(
                        prepared.state)
                  : conditioned;
  RouteObservation warm{};
  const rund::compute::Status repeated =
      samples ? RunThroughDeviceVsmProductRoute(prepared.state, warm) : samples;
  const rund::compute::Status ended =
      repeated
          ? rund::compute::detail::end_virtual_pipeline_samples(prepared.state)
          : repeated;

  std::uint64_t queue_after = 0u;
  const PublicationSnapshot after_primary =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot after_alternate =
      SnapshotPublication(prepared.state->alternate_pipeline);
  const std::uint64_t version_after = BackingVersion(*prepared.output);
  const std::uint64_t recovery_after = BackingRecovery(*prepared.output);
  const std::uint64_t bytes = prepared.expected.size() * sizeof(std::uint32_t);
  const rund::compute::ResidencyStats &stats =
      prepared.state->stats.pipeline.residency;
  const bool valid =
      ended && queue_counter(prepared.state, queue_after) &&
      queue_after == queue_before + 2u && ExactOutput(prepared) &&
      ExactDeviceVsmEvidence(cold, pages, pages, bytes, bytes, 0u) &&
      ExactDeviceVsmEvidence(warm, pages, pages, bytes, bytes, 0u) &&
      cold.evidence.cold_prepare_count == 1u &&
      cold.evidence.warm_rearm_count == 0u &&
      warm.evidence.cold_prepare_count == 1u &&
      warm.evidence.warm_rearm_count == 1u &&
      after_primary.generation == before_primary.generation + 2u &&
      after_primary.payload_epoch == before_primary.payload_epoch + 2u &&
      after_alternate.generation == before_alternate.generation + 2u &&
      after_alternate.payload_epoch == before_alternate.payload_epoch + 2u &&
      version_after == version_before + 2u && recovery_after == 0u &&
      stats.sampled_runs == 1u && stats.allocation_free_runs == 1u;
  if (!valid) {
    std::fprintf(
        stderr,
        "DeviceVsm warm backend=%u status=%u queue=%llu/%llu cold=%llu/%llu "
        "warm=%llu/%llu samples=%u/%u publication=%llu/%llu "
        "version=%llu/%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(ended.ok()),
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_after),
        static_cast<unsigned long long>(cold.evidence.cold_prepare_count),
        static_cast<unsigned long long>(cold.evidence.warm_rearm_count),
        static_cast<unsigned long long>(warm.evidence.cold_prepare_count),
        static_cast<unsigned long long>(warm.evidence.warm_rearm_count),
        stats.sampled_runs, stats.allocation_free_runs,
        static_cast<unsigned long long>(before_primary.generation),
        static_cast<unsigned long long>(after_primary.generation),
        static_cast<unsigned long long>(version_before),
        static_cast<unsigned long long>(version_after));
  } else {
    std::fprintf(stderr,
                 "DeviceVsm warm backend=%u cold=1 rearm=1 samples=1/1 "
                 "queue=2 publication=2 backing=2\n",
                 static_cast<unsigned>(backend));
  }
  return valid;
}

} // namespace rund_node_test_device_vsm_product

#endif
