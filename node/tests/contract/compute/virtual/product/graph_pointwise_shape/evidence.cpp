#include "evidence.hpp"

#include "src/compute/device/residency/pool.hpp"
#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"
#include "src/compute/virtual/state.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <mutex>
#include <vector>

namespace rund_node_test_virtual::product::graph_pointwise_shape {
namespace {

using Owner =
    rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;

[[nodiscard]] std::shared_ptr<Owner> owner(const EvidenceView &view) noexcept {
  return view.state == nullptr ? std::shared_ptr<Owner>{}
                               : std::static_pointer_cast<Owner>(
                                     view.state->device_vsm_product_cache);
}

[[nodiscard]] bool proof_valid(const EvidenceView &view,
                               const Owner &value) noexcept {
  using namespace rund::node::accel::detail;
  const auto &proof = value.proof;
  if (proof == nullptr ||
      proof->topology != DeviceVsmTopology::GraphPointwise ||
      proof->plan.input_buffer_count != view.input_count ||
      proof->plan.output_buffer_count != 1u ||
      proof->residents.input_count != view.input_count ||
      proof->residents.output_count != 1u ||
      proof->residents.count != view.input_count + 1u ||
      proof->graph_pointwise.stage_count != view.stage_count ||
      proof->graph_pointwise.topology.external_input_count !=
          view.input_count ||
      proof->graph_pointwise.topology.stages[0u].input_count !=
          view.input_count) {
    return false;
  }
  for (std::size_t input = 0u; input < view.input_count; ++input) {
    const DeviceVsmGraphValueSource source =
        proof->graph_pointwise.topology.stages[0u].inputs[input];
    if (source.kind != DeviceVsmGraphValueSourceKind::ExternalInput ||
        source.index != input) {
      return false;
    }
  }
  for (std::size_t stage = 1u; stage < view.stage_count; ++stage) {
    if (proof->graph_pointwise.topology.stages[stage].input_count != 1u) {
      return false;
    }
    const DeviceVsmGraphValueSource source =
        proof->graph_pointwise.topology.stages[stage].inputs[0u];
    if (source.kind != DeviceVsmGraphValueSourceKind::StageOutput ||
        source.index + 1u != stage) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool native_valid(const EvidenceView &view,
                                const Owner &value) noexcept {
  using namespace rund::node::accel::detail;
  const auto &evidence = value.evidence;
  std::uint32_t expected_reuse = 0u;
  std::uint32_t expected_checksum = 0u;
  std::uint32_t expected_round_trips = 0u;
  std::uint64_t expected_state_bytes = 0u;
  std::uint64_t expected_scratch_bytes = 0u;
  if (value.proof == nullptr ||
      !device_vsm_ring_schedule_expected(value.proof->geometry,
                                         value.proof->width, expected_reuse,
                                         expected_checksum) ||
      !device_vsm_ring_storage_expected(
          value.proof->geometry, value.proof->width,
          value.proof->residents.count, expected_state_bytes,
          expected_scratch_bytes) ||
      !device_vsm_ring_round_trips_expected(value.proof->geometry,
                                            value.proof->residents.count,
                                            expected_round_trips)) {
    return false;
  }
  return evidence != nullptr && evidence->public_handoff_count == 1u &&
         evidence->authority_accept_count == 1u &&
         evidence->pipeline_terminal_count == view.stage_count &&
         evidence->backing_publication_count == 1u &&
         evidence->final_received && !evidence->quarantined &&
         evidence->native.page_count == view.page_count &&
         evidence->native.generated_epochs == view.page_count &&
         evidence->native.completed_epochs == view.page_count &&
         evidence->native.ring_reuse_transitions == expected_reuse &&
         evidence->native.ring_schedule_checksum == expected_checksum &&
         evidence->native.ring_round_trips == expected_round_trips &&
         evidence->native.ring_state_bytes == expected_state_bytes &&
         evidence->native.ring_scratch_bytes == expected_scratch_bytes &&
         evidence->native.native_submit_count == 1u &&
         evidence->native.epoch_native_submit_count == 0u &&
         evidence->native.payload_dispatch_count == 1u &&
         evidence->native.host_service_turn_count == 0u &&
         evidence->native.host_epoch_callback_count == 0u &&
         evidence->native.final_callback_count == 1u;
}

[[nodiscard]] bool backing_valid(const EvidenceView &view,
                                 const std::uint64_t initial_version,
                                 const std::size_t logical_bytes) noexcept {
  using rund::compute::detail::VirtualBackingAccess;
  for (const auto &backing : view.input_backings) {
    if (backing == nullptr) {
      return false;
    }
    const BackingFacts facts = backing->facts();
    if (facts.read_count != view.page_count ||
        facts.read_bytes != logical_bytes) {
      return false;
    }
  }
  if (view.output_backing == nullptr) {
    return false;
  }
  const BackingFacts output = view.output_backing->facts();
  return output.write_count == view.page_count &&
         output.write_bytes == logical_bytes &&
         VirtualBackingAccess::version(*view.output_backing) ==
             initial_version + 1u &&
         VirtualBackingAccess::recovery_bytes(*view.output_backing) == 0u;
}

[[nodiscard]] bool output_valid(const EvidenceView &view) noexcept {
  if (view.output_backing == nullptr) {
    return false;
  }
  std::vector<std::uint64_t> observed(view.expected.size());
  return view.output_backing->observe(
             std::as_writable_bytes(std::span{observed})) &&
         std::equal(observed.begin(), observed.end(), view.expected.begin(),
                    view.expected.end()) &&
         view.output_backing->tail_poisoned();
}

} // namespace

void stage_generations(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState> &state,
    const std::span<std::uint64_t> result) noexcept {
  using rund::compute::detail::residency::Pool;
  if (state == nullptr ||
      state->graph_pipelines.size() != result.size() * Pool::BankCount) {
    std::fill(result.begin(), result.end(),
              std::numeric_limits<std::uint64_t>::max());
    return;
  }
  for (std::size_t stage = 0u; stage < result.size(); ++stage) {
    const auto &pipeline = state->graph_pipelines[stage * Pool::BankCount];
    if (pipeline == nullptr || pipeline->publication == nullptr) {
      std::fill(result.begin(), result.end(),
                std::numeric_limits<std::uint64_t>::max());
      return;
    }
    std::lock_guard state_lock{pipeline->gate};
    std::lock_guard publication_lock{pipeline->publication->gate};
    result[stage] = pipeline->publication->generation;
  }
}

bool validate(
    const EvidenceView &view, const rund::compute::Backend backend,
    const rund::compute::Status &status, const std::uint64_t initial_version,
    const std::span<const std::uint64_t> initial_generations) noexcept {
  using namespace rund::compute;
  if (view.input_count == 0u || view.stage_count == 0u ||
      view.frame_elements == 0u || view.page_count == 0u ||
      view.tail_elements >= view.page_count * view.frame_elements ||
      view.input_backings.size() != view.input_count ||
      initial_generations.size() != view.stage_count) {
    return false;
  }
  const std::size_t element_count =
      view.page_count * view.frame_elements - view.tail_elements;
  const std::size_t logical_bytes = element_count * sizeof(std::uint64_t);
  if (view.expected.size() != element_count) {
    return false;
  }
  const Stats &stats = view.stats;
  const ResidencyStats residency = stats.pipeline.residency;
  const std::shared_ptr<Owner> retained = owner(view);
  std::vector<std::uint64_t> final_generations(view.stage_count);
  stage_generations(view.state, final_generations);
  bool generations_valid = true;
  for (std::size_t stage = 0u; stage < view.stage_count; ++stage) {
    generations_valid =
        generations_valid &&
        initial_generations[stage] <
            std::numeric_limits<std::uint64_t>::max() - 1u &&
        final_generations[stage] == initial_generations[stage] + 1u;
  }
  const bool valid =
      backend != Backend::Cpu && status && retained != nullptr &&
      proof_valid(view, *retained) && native_valid(view, *retained) &&
      backing_valid(view, initial_version, logical_bytes) &&
      generations_valid && view.frame_capacity == 2u &&
      residency.window_handoff_count == 1u &&
      residency.window_batch_count == 1u &&
      residency.window_queue_call_count == 1u &&
      residency.page_in_count == view.input_count * view.page_count &&
      residency.backing_read_bytes == view.input_count * logical_bytes &&
      residency.page_out_count == view.page_count &&
      residency.backing_write_bytes == logical_bytes &&
      stats.command_submits == 1u && stats.dispatches == 1u &&
      stats.final_dispatches == 1u && output_valid(view);
  std::fprintf(stderr,
               "Graph %s GPU backend=%u valid=%u owner=%u submit=%llu "
               "dispatch=%llu page=%llu/%llu read=%llu final=%llu "
               "publish=%llu/%llu/%llu\n",
               view.label == nullptr ? "shape" : view.label,
               static_cast<unsigned>(backend), static_cast<unsigned>(valid),
               static_cast<unsigned>(retained != nullptr),
               static_cast<unsigned long long>(stats.command_submits),
               static_cast<unsigned long long>(stats.dispatches),
               static_cast<unsigned long long>(residency.page_in_count),
               static_cast<unsigned long long>(residency.page_out_count),
               static_cast<unsigned long long>(residency.backing_read_bytes),
               retained == nullptr || retained->evidence == nullptr
                   ? 0u
                   : static_cast<unsigned long long>(
                         retained->evidence->native.final_callback_count),
               retained == nullptr || retained->evidence == nullptr
                   ? 0u
                   : static_cast<unsigned long long>(
                         retained->evidence->authority_accept_count),
               retained == nullptr || retained->evidence == nullptr
                   ? 0u
                   : static_cast<unsigned long long>(
                         retained->evidence->pipeline_terminal_count),
               retained == nullptr || retained->evidence == nullptr
                   ? 0u
                   : static_cast<unsigned long long>(
                         retained->evidence->backing_publication_count));
  return valid;
}

} // namespace rund_node_test_virtual::product::graph_pointwise_shape
