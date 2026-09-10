#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../evidence.hpp"
#include "../fixture.hpp"
#include "../route.hpp"

#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/state.hpp"

#include <cstdio>

namespace rund_node_test_persistent_product::window_test {

namespace {

struct StepInfo final {
  unsigned route{};
  unsigned window{};
  unsigned job_prepared{};
};

[[nodiscard]] StepInfo step_info(const rund::compute::detail::PipelineState *state,
                                 const std::size_t index) noexcept {
  if (state == nullptr || index >= state->steps.size()) {
    return {};
  }
  const auto &step = state->steps[index];
  return {static_cast<unsigned>(step.route),
          static_cast<unsigned>(step.window),
          static_cast<unsigned>(step.job != nullptr && step.job->prepared.ok)};
}

[[nodiscard]] bool ExactWindowPublication(
    const PublicationSnapshot before_primary,
    const PublicationSnapshot after_primary,
    const PublicationSnapshot before_alternate,
    const PublicationSnapshot after_alternate,
    const std::uint64_t version_before, const std::uint64_t version_after,
    const std::uint64_t recovery_after,
    const std::uint64_t coordinates) noexcept {
  const std::uint64_t primary_runs = (coordinates + 1u) / 2u;
  const std::uint64_t alternate_runs = coordinates / 2u;
  return after_primary.generation ==
             before_primary.generation + primary_runs &&
         after_primary.payload_epoch ==
             before_primary.payload_epoch + primary_runs &&
         after_alternate.generation ==
             before_alternate.generation + alternate_runs &&
         after_alternate.payload_epoch ==
             before_alternate.payload_epoch + alternate_runs &&
         version_after == version_before + 1u && recovery_after == 0u;
}

} // namespace

bool CheckPersistentWindowFallback(const rund::compute::Backend backend,
                                   const NativeQueueCounter queue_counter,
                                   const std::uint64_t coordinates,
                                   const Edge edge,
                                   bool &unavailable) noexcept {
  WindowPrepared prepared{};
  if (!PrepareWindowProduct(backend, coordinates, edge, prepared,
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
  const rund::compute::detail::PipelineState *const pipeline =
      prepared.state == nullptr ? nullptr : prepared.state->pipeline.get();
  const std::size_t step_count =
      pipeline == nullptr ? 0u : pipeline->steps.size();
  const StepInfo first_step = step_info(pipeline, 0u);
  const StepInfo second_step = step_info(pipeline, 1u);
  ProductRouteObservation observation{};
  const rund::compute::Status status =
      RunThroughPersistentProductRoute(prepared.state, observation);
  const std::uint64_t queue_calls =
      backend == rund::compute::Backend::Vulkan ? 1u : coordinates;
  const std::uint64_t native_batches = coordinates;
  std::uint64_t queue_after = 0u;
  const bool queue_exact = queue_counter(prepared.state, queue_after) &&
                           queue_after == queue_before + queue_calls;
  const PublicationSnapshot after_primary =
      SnapshotPublication(prepared.state->pipeline);
  const PublicationSnapshot after_alternate =
      SnapshotPublication(prepared.state->alternate_pipeline);
  const std::uint64_t version_after = BackingVersion(*prepared.output);
  const std::uint64_t recovery_after = BackingRecovery(*prepared.output);
  const std::uint64_t pages = ProductPages(coordinates, edge);
  const std::uint64_t logical_elements = ProductElements(coordinates, edge);
  const std::uint64_t backing_bytes = logical_elements * sizeof(std::uint32_t);
  const std::uint64_t frame_bytes =
      pages * FrameElements * sizeof(std::uint32_t);
  const rund::compute::Stats &run_stats = prepared.state->stats;
  const rund::compute::ResidencyStats &stats =
      prepared.state->stats.pipeline.residency;
  const bool sliding_fallback =
      observation.production_route &&
      observation.prepare_status.reason() ==
          rund::compute::Reason::BackendUnsupported &&
      !observation.prepared && observation.owner == nullptr &&
      !observation.mode_snapshot && !observation.final_received;
  const bool valid =
      status && sliding_fallback && queue_exact && ExactWindowOutput(prepared) &&
      ExactWindowPublication(before_primary, after_primary, before_alternate,
                             after_alternate, version_before, version_after,
                             recovery_after, coordinates) &&
      stats.window_handoff_count == 1u &&
      stats.window_batch_count == native_batches &&
      stats.window_queue_call_count == queue_calls &&
      stats.backing_read_bytes == backing_bytes &&
      stats.page_in_bytes == frame_bytes &&
      stats.page_in_count == pages && stats.cache_hit_count == 0u &&
      run_stats.command_submits == queue_calls &&
      run_stats.publication.generation == after_primary.generation;
  if (!valid) {
    std::fprintf(
        stderr,
        "persistent Window backend=%u edge=%u Q=%llu status=%u queue=%llu/%llu "
        "reason=%u steps=%zu first=%u/%u/%u second=%u/%u/%u "
        "prepare=%u/%u/%u final=%u fallback=%u gpu=%llu submit=%llu "
        "epoch=%llu host=%llu "
        "read=%llu/%llu promoted=%llu/%llu calls=%llu hits=%llu "
        "output=%u command=%llu pub_gen=%llu pub_commits=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(edge),
        static_cast<unsigned long long>(coordinates),
        static_cast<unsigned>(static_cast<bool>(status)),
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_after),
        static_cast<unsigned>(status.reason()), step_count,
        first_step.route, first_step.window, first_step.job_prepared,
        second_step.route, second_step.window, second_step.job_prepared,
        static_cast<unsigned>(static_cast<bool>(observation.prepare_status)),
        static_cast<unsigned>(observation.prepare_status.reason()),
        static_cast<unsigned>(observation.prepared),
        static_cast<unsigned>(observation.final_received),
        static_cast<unsigned>(sliding_fallback),
        static_cast<unsigned long long>(
            observation.final.evidence.gpu_completed_coordinates),
        static_cast<unsigned long long>(
            observation.final.evidence.native_submit_count),
        static_cast<unsigned long long>(
            observation.final.evidence.epoch_native_submit_count),
        static_cast<unsigned long long>(
            observation.final.evidence.host_epoch_callback_count),
        static_cast<unsigned long long>(stats.backing_read_bytes),
        static_cast<unsigned long long>(backing_bytes),
        static_cast<unsigned long long>(stats.page_in_bytes),
        static_cast<unsigned long long>(frame_bytes),
        static_cast<unsigned long long>(stats.page_in_count),
        static_cast<unsigned long long>(stats.cache_hit_count),
        static_cast<unsigned>(ExactWindowOutput(prepared)),
        static_cast<unsigned long long>(run_stats.command_submits),
        static_cast<unsigned long long>(run_stats.publication.generation),
        static_cast<unsigned long long>(run_stats.publication.commit_count));
  }
  return valid;
}

} // namespace rund_node_test_persistent_product::window_test

namespace rund_node_test_persistent_product {

bool CheckPersistentWindowProduct(const rund::compute::Backend backend,
                                  const NativeQueueCounter queue_counter,
                                  bool &unavailable) noexcept {
  return window_test::CheckPersistentWindowFallback(
             backend, queue_counter, 2u, window_test::Edge::Clamp,
             unavailable) &&
         (unavailable ||
          window_test::CheckPersistentWindowFallback(
              backend, queue_counter, 5u, window_test::Edge::Clamp,
              unavailable));
}

} // namespace rund_node_test_persistent_product

#else

namespace rund_node_test_persistent_product {

bool CheckPersistentWindowProduct(const rund::compute::Backend,
                                  const NativeQueueCounter, bool &) noexcept {
  return true;
}

} // namespace rund_node_test_persistent_product

#endif
