#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../persistent_product/fixture.hpp"
#include "../evidence.hpp"

#include "src/accel/kernel/prepared/model.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <array>
#include <cstdio>

namespace rund_node_test_device_vsm_product::graph_test {
namespace {

using PrepareGraphCase = bool (*)(rund::compute::Backend, std::uint64_t,
                                  PreparedGraph &, bool &);

struct GraphVariant final {
  const char *name{};
  PrepareGraphCase prepare{};
  std::uint64_t authored_nodes{};
  std::uint64_t lowered_nodes{};
};

[[nodiscard]] bool
ExactGraphWavefront(const rund::compute::detail::device_vsm_product_detail::
                        DeviceVsmProductOwner *const owner,
                    const rund::node::accel::detail::DeviceVsmEvidence &native,
                    const std::uint64_t pages, const std::uint64_t stages,
                    const std::uint64_t frame_capacity) noexcept {
  if (owner == nullptr || owner->proof == nullptr || stages < 2u ||
      frame_capacity == 0u || frame_capacity > pages) {
    return false;
  }
  const auto &wavefront = owner->proof->graph_map_reduce.wavefront;
  std::uint32_t expected_steps = 0u;
  std::uint32_t expected_trace = 0u;
  return wavefront.stage_count == stages && wavefront.map_stage == 0u &&
         wavefront.collective_stage == stages - 1u &&
         wavefront.frame_capacity == frame_capacity &&
         wavefront.batch_count ==
             pages / frame_capacity +
                 static_cast<std::uint64_t>(pages % frame_capacity != 0u) &&
         rund::node::accel::detail::device_vsm_graph_wavefront_expected(
             wavefront, expected_steps, expected_trace) &&
         native.graph_wavefront_steps == expected_steps &&
         native.graph_wavefront_trace == expected_trace;
}

[[nodiscard]] bool RunGraphOverflowCase(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    bool &unavailable) noexcept {
  using namespace rund_node_test_persistent_product;
  PreparedGraph prepared{};
  if (!PrepareGraphOverflow(backend, 5u, prepared, unavailable)) {
    return unavailable;
  }
  const std::shared_ptr<rund::compute::detail::PipelineState> primary =
      prepared.state->device_vsm_semantic_pipeline != nullptr
          ? prepared.state->device_vsm_semantic_pipeline
          : prepared.state->pipeline;
  const PublicationSnapshot before_map = SnapshotPublication(primary);
  const PublicationSnapshot before_sum =
      SnapshotPublication(rund::compute::detail::graph_terminal_pipeline(
          *prepared.state, 0u));
  const std::uint64_t version_before = BackingVersion(*prepared.output);
  const std::uint64_t recovery_before = BackingRecovery(*prepared.output);
  std::uint64_t queue_before = 0u;
  if (queue_counter == nullptr ||
      !queue_counter(prepared.state, queue_before)) {
    return false;
  }
  RouteObservation observation{};
  const rund::compute::Status status =
      RunThroughDeviceVsmProductRoute(prepared.state, observation);
  std::uint64_t queue_after = 0u;
  const auto &native = observation.evidence.native;
  const auto owner = std::static_pointer_cast<
      rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner>(
      observation.owner);
  const std::uint64_t frame_capacity =
      prepared.state->pipeline->residency->tiled_graph().frame_capacity();
  const PublicationSnapshot after_map = SnapshotPublication(primary);
  const PublicationSnapshot after_sum =
      SnapshotPublication(rund::compute::detail::graph_terminal_pipeline(
          *prepared.state, 0u));
  const bool valid =
      !status && status.reason() == rund::compute::Reason::ReduceSumOverflow &&
      queue_counter(prepared.state, queue_after) &&
      queue_after == queue_before + 1u && observation.production_route &&
      observation.prepared && observation.executed &&
      observation.evidence.final_received &&
      !observation.evidence.quarantined && native.page_count == 5u &&
      native.generated_epochs == 5u && native.completed_epochs == 5u &&
      ExactGraphWavefront(owner.get(), native, 5u, prepared.stage_count,
                          frame_capacity) &&
      native.forecasted_pages == 5u && native.promoted_pages == 5u &&
      native.drained_pages == 0u && native.persisted_pages == 0u &&
      native.gpu_backing_read_bytes == prepared.input_bytes &&
      native.gpu_backing_write_bytes == 0u &&
      native.native_submit_count == 1u &&
      native.epoch_native_submit_count == 0u &&
      native.host_service_turn_count == 0u &&
      native.host_epoch_callback_count == 0u &&
      native.final_callback_count == 1u && native.may_write &&
      after_map.generation == before_map.generation &&
      after_map.payload_epoch == before_map.payload_epoch &&
      after_sum.generation == before_sum.generation &&
      after_sum.payload_epoch == before_sum.payload_epoch &&
      BackingVersion(*prepared.output) == version_before &&
      BackingRecovery(*prepared.output) == recovery_before;
  if (valid) {
    std::fprintf(
        stderr,
        "DeviceVsm Graph overflow backend=%u Q=5 submit=%llu "
        "epoch_submit=%llu host_turn=%llu host_callback=%llu final=%llu "
        "generated=%llu completed=%llu publish=0\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned long long>(native.native_submit_count),
        static_cast<unsigned long long>(native.epoch_native_submit_count),
        static_cast<unsigned long long>(native.host_service_turn_count),
        static_cast<unsigned long long>(native.host_epoch_callback_count),
        static_cast<unsigned long long>(native.final_callback_count),
        static_cast<unsigned long long>(native.generated_epochs),
        static_cast<unsigned long long>(native.completed_epochs));
  }
  return valid;
}

[[nodiscard]] bool RunGraphCase(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter queue_counter,
    const GraphVariant variant, const std::uint64_t pages, bool &unavailable,
    std::uint64_t &retained) noexcept {
  using namespace rund_node_test_persistent_product;
  PreparedGraph prepared{};
  if (variant.prepare == nullptr ||
      !variant.prepare(backend, pages, prepared, unavailable)) {
    return unavailable;
  }
  const std::shared_ptr<rund::compute::detail::PipelineState> primary =
      prepared.state->device_vsm_semantic_pipeline != nullptr
          ? prepared.state->device_vsm_semantic_pipeline
          : prepared.state->pipeline;
  const PublicationSnapshot before_map = SnapshotPublication(primary);
  const PublicationSnapshot before_sum =
      SnapshotPublication(rund::compute::detail::graph_terminal_pipeline(
          *prepared.state, 0u));
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
  const PublicationSnapshot after_map = SnapshotPublication(primary);
  const PublicationSnapshot after_sum =
      SnapshotPublication(rund::compute::detail::graph_terminal_pipeline(
          *prepared.state, 0u));
  const std::uint64_t version_after = BackingVersion(*prepared.output);
  const std::uint64_t recovery_after = BackingRecovery(*prepared.output);
  const auto owner = std::static_pointer_cast<
      rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner>(
      observation.owner);
  const std::uint64_t frame_capacity =
      prepared.state->pipeline->residency->tiled_graph().frame_capacity();
  retained =
      owner == nullptr ? 0u : owner->preparation.capability.retained_bytes;
  const rund::compute::ResidencyStats &stats =
      prepared.state->stats.pipeline.residency;
  const bool valid =
      status && queue_counter(prepared.state, queue_after) &&
      queue_after == queue_before + 1u && ExactGraphOutput(prepared) &&
      owner != nullptr && owner->proof != nullptr &&
      owner->proof->topology ==
          rund::node::accel::detail::DeviceVsmTopology::GraphMapReduce &&
      ExactGraphWavefront(owner.get(), observation.evidence.native, pages,
                          prepared.stage_count, frame_capacity) &&
      owner->proof->graph_map_reduce.semantic.op == prepared.operation &&
      prepared.authored_nodes == variant.authored_nodes &&
      prepared.lowered_nodes == variant.lowered_nodes &&
      owner->proof->output_bytes == sizeof(std::uint64_t) &&
      ExactDeviceVsmEvidence(observation, pages, 1u, prepared.input_bytes,
                             sizeof(std::uint64_t), 0u) &&
      ExactPublication(before_map, after_map, before_sum, after_sum,
                       version_before, version_after, recovery_after) &&
      stats.window_handoff_count == 1u && stats.window_batch_count == 1u &&
      stats.window_queue_call_count == 1u && stats.page_in_count == pages &&
      stats.page_out_count == 1u &&
      stats.backing_read_bytes == prepared.input_bytes &&
      stats.backing_write_bytes == sizeof(std::uint64_t) &&
      stats.page_in_bytes == prepared.input_bytes &&
      stats.page_out_bytes == sizeof(std::uint64_t);
  if (valid) {
    const auto &native = observation.evidence.native;
    std::fprintf(
        stderr,
        "DeviceVsm Graph %s backend=%u Q=%llu submit=%llu "
        "epoch_submit=%llu "
        "host_turn=%llu host_callback=%llu generated=%llu completed=%llu "
        "final=%llu pipeline=2 backing=1\n",
        variant.name, static_cast<unsigned>(backend),
        static_cast<unsigned long long>(pages),
        static_cast<unsigned long long>(native.native_submit_count),
        static_cast<unsigned long long>(native.epoch_native_submit_count),
        static_cast<unsigned long long>(native.host_service_turn_count),
        static_cast<unsigned long long>(native.host_epoch_callback_count),
        static_cast<unsigned long long>(native.generated_epochs),
        static_cast<unsigned long long>(native.completed_epochs),
        static_cast<unsigned long long>(native.final_callback_count));
  }
  if (!valid) {
    const auto *const native =
        static_cast<const rund::node::accel::detail::prepared::PipelineState *>(
            prepared.state->pipeline->prepared.owner.get());
    const auto *const run = native == nullptr || native->state_count == 0u ||
                                    native->states == nullptr ||
                                    native->states[0u] == nullptr
                                ? nullptr
                                : native->states[0u].get();
    const auto *const step = run == nullptr ||
                                     run->bound.run.step_count == 0u ||
                                     run->bound.run.steps == nullptr
                                 ? nullptr
                                 : &run->bound.run.steps[0u];
    std::fprintf(
        stderr,
        "DeviceVsm Graph %s backend=%u Q=%llu status=%u route=%u/%u/%u "
        "queue=%llu/%llu topology=%u final=%u counts=%llu/%llu/%llu/%llu "
        "bytes=%llu/%llu output=%u reason=%s native=%llu/%llu "
        "bound=%u/%llu/%u/%u step=%u/%u/%u/%u\n",
        variant.name, static_cast<unsigned>(backend),
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
        static_cast<unsigned>(observation.evidence.final_received),
        static_cast<unsigned long long>(
            observation.evidence.native.generated_epochs),
        static_cast<unsigned long long>(
            observation.evidence.native.completed_epochs),
        static_cast<unsigned long long>(
            observation.evidence.native.drained_pages),
        static_cast<unsigned long long>(
            observation.evidence.native.persisted_pages),
        static_cast<unsigned long long>(stats.backing_read_bytes),
        static_cast<unsigned long long>(stats.backing_write_bytes),
        static_cast<unsigned>(ExactGraphOutput(prepared)),
        observation.prepare_reason == nullptr ? "null"
                                              : observation.prepare_reason,
        native == nullptr
            ? 0ull
            : static_cast<unsigned long long>(native->state_count),
        native == nullptr ? 0ull
                          : static_cast<unsigned long long>(native->size),
        static_cast<unsigned>(run != nullptr && run->bound.ok),
        run == nullptr
            ? 0ull
            : static_cast<unsigned long long>(run->bound.run.step_count),
        static_cast<unsigned>(run != nullptr &&
                              run->bound.run.steps != nullptr),
        static_cast<unsigned>(run != nullptr &&
                              (run->bound.run.resets == nullptr ||
                               run->bound.run.resets->empty())),
        static_cast<unsigned>(step != nullptr && step->step != nullptr),
        static_cast<unsigned>(step != nullptr && step->planned != nullptr),
        static_cast<unsigned>(step != nullptr && step->control.active()),
        static_cast<unsigned>(step != nullptr && step->step != nullptr &&
                              step->step->map_semantic.recurrence_total));
  }
  return valid;
}

} // namespace

bool RunGraphProductCases(
    const rund::compute::Backend backend,
    const rund_node_test_persistent_product::NativeQueueCounter
        queue_counter) noexcept {
  if (!CheckFusedGraphSemantic()) {
    return false;
  }
  const std::array variants{
      GraphVariant{.name = "map-sum",
                   .prepare = PrepareGraphProduct,
                   .authored_nodes = 2u,
                   .lowered_nodes = 2u},
      GraphVariant{.name = "fused-map-prefix-sum",
                   .prepare = PrepareFusedGraphProduct,
                   .authored_nodes = 3u,
                   .lowered_nodes = 2u},
      GraphVariant{.name = "typed-map-sum",
                   .prepare = PrepareTypedGraphProduct,
                   .authored_nodes = 2u,
                   .lowered_nodes = 2u},
      GraphVariant{.name = "fused-typed-map-prefix-sum",
                   .prepare = PrepareFusedTypedGraphProduct,
                   .authored_nodes = 3u,
                   .lowered_nodes = 2u},
      GraphVariant{.name = "public-input-branch-sum",
                   .prepare = PrepareBranchGraphProduct,
                   .authored_nodes = 4u,
                   .lowered_nodes = 4u},
      GraphVariant{.name = "map-count-nonzero",
                   .prepare = PrepareGraphCountNonzero,
                   .authored_nodes = 2u,
                   .lowered_nodes = 2u},
      GraphVariant{.name = "map-min",
                   .prepare = PrepareGraphMin,
                   .authored_nodes = 2u,
                   .lowered_nodes = 2u},
      GraphVariant{.name = "map-max",
                   .prepare = PrepareGraphMax,
                   .authored_nodes = 2u,
                   .lowered_nodes = 2u},
  };
  for (const GraphVariant variant : variants) {
    std::array<std::uint64_t, 3u> retained{};
    std::size_t index = 0u;
    for (const std::uint64_t pages : {5u, 9u, 257u}) {
      bool unavailable = false;
      if (!RunGraphCase(backend, queue_counter, variant, pages, unavailable,
                        retained[index++])) {
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
  bool unavailable = false;
  return RunGraphOverflowCase(backend, queue_counter, unavailable);
}

} // namespace rund_node_test_device_vsm_product::graph_test

#endif
