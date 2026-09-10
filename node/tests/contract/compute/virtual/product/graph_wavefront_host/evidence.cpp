#include "internal.hpp"

#include "../route.hpp"

#include "src/compute/device/residency/pool.hpp"
#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/state.hpp"
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "src/accel/vulkan/kernel/pipeline/residency/local.hpp"
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace rund_node_test_virtual::product::graph_wavefront_host {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool check_graph_admission(
    const rund::compute::detail::VirtualPipelineState &state) noexcept {
  using namespace rund::node::accel::detail;
  constexpr std::size_t BankCount = 2u;
  if (state.graph_pipelines.size() != StageCount * BankCount) {
    return false;
  }
  for (const auto &graph_pipeline : state.graph_pipelines) {
    if (graph_pipeline == nullptr) {
      return false;
    }
    VulkanResidencyAdmissionSnapshot snapshot{};
    if (!InspectVulkanResidencyAdmission(graph_pipeline->prepared, snapshot) ||
        !snapshot.final_selected ||
        snapshot.final_candidate !=
            VulkanResidencyAdmissionCandidate::GraphStageGeneratedIndirect ||
        snapshot.selected_mode !=
            VulkanResidencyMode::GraphStageGeneratedIndirect ||
        snapshot.generation_tag == 0u || snapshot.candidate_count < 2u) {
      return false;
    }
    const VulkanResidencyAdmissionCandidateSnapshot *candidate = nullptr;
    for (std::size_t index = 0u;
         index < snapshot.candidate_count && index < snapshot.candidates.size();
         ++index) {
      if (snapshot.candidates[index].candidate ==
          VulkanResidencyAdmissionCandidate::GraphStageGeneratedIndirect) {
        candidate = &snapshot.candidates[index];
        break;
      }
    }
    if (candidate == nullptr ||
        candidate->candidate !=
            VulkanResidencyAdmissionCandidate::GraphStageGeneratedIndirect ||
        candidate->reason !=
            VulkanResidencyAdmissionReason::PredicateAccepted ||
        candidate->generation_tag != snapshot.generation_tag ||
        candidate->entry_count != 2u || candidate->active_step_count != 2u) {
      return false;
    }
  }
  return true;
}

bool check_graph_replay(
    const rund::compute::detail::VirtualPipelineState &state,
    const std::uint64_t capacity, const std::uint64_t batches) noexcept {
  using namespace rund::node::accel::detail;
  constexpr std::size_t BankCount = 2u;
  if (state.graph_pipelines.size() != StageCount * BankCount ||
      capacity != 2u || batches != 3u) {
    return false;
  }
  for (std::size_t index = 0u; index < state.graph_pipelines.size(); ++index) {
    const auto &graph_pipeline = state.graph_pipelines[index];
    if (graph_pipeline == nullptr) {
      return false;
    }
    VulkanResidencyGraphGeneratedDiagnostics diagnostics{};
    if (!InspectVulkanGraphGeneratedDiagnostics(graph_pipeline->prepared,
                                                diagnostics)) {
      return false;
    }
    const std::size_t bank = index % BankCount;
    const std::uint64_t expected_seq =
        (batches + BankCount - 1u - bank) / BankCount;
    const std::uint64_t full_batches = GraphPageCount / capacity;
    const std::uint64_t tail_active = GraphPageCount % capacity;
    const std::uint64_t active_rows =
        full_batches + (bank < tail_active ? 1u : 0u);
    const bool cold_identity =
        diagnostics.generation != 0u &&
        diagnostics.record_generation == diagnostics.generation &&
        diagnostics.admission_count == capacity &&
        diagnostics.ready_count == capacity &&
        diagnostics.recorded_count == capacity;
    const bool idle =
        diagnostics.phase == VulkanResidencyGraphGeneratedPhase::Recorded &&
        diagnostics.submission_generation == 0u &&
        diagnostics.submitted_count == 0u && diagnostics.terminal_count == 0u &&
        diagnostics.accepted_count == 0u &&
        diagnostics.known_failure_count == 0u &&
        diagnostics.unknown_count == 0u && !diagnostics.failed &&
        !diagnostics.replay && !diagnostics.generation_overflow &&
        !diagnostics.quarantined && diagnostics.first_reason != nullptr &&
        std::strcmp(diagnostics.first_reason, "ok") == 0;
    const bool cycle_totals = diagnostics.submit_seq == expected_seq &&
                              diagnostics.done_seq == diagnostics.submit_seq &&
                              diagnostics.submit_total == active_rows &&
                              diagnostics.done_total == active_rows &&
                              diagnostics.accept_total == active_rows &&
                              diagnostics.known_total == 0u &&
                              diagnostics.unknown_total == 0u;
    if (!cold_identity || !idle || !cycle_totals) {
      return false;
    }
  }
  return true;
}
#else
bool check_graph_admission(
    const rund::compute::detail::VirtualPipelineState &) noexcept {
  return true;
}

bool check_graph_replay(const rund::compute::detail::VirtualPipelineState &,
                        std::uint64_t, std::uint64_t) noexcept {
  return true;
}
#endif

bool validate_case(
    Case &test_case, const rund::compute::Backend backend,
    const rund::compute::Status &first_run, const std::uint64_t initial_version,
    const rund_node_test_virtual::product::ProductRouteObservation
        &route_observation) noexcept {
  using namespace rund::compute;
  const Stats first_stats = test_case.pipeline.stats();
  const ResidencyStats first_residency = first_stats.pipeline.residency;
  const BackingFacts first_facts = test_case.first_backing->facts();
  const BackingFacts second_facts = test_case.second_backing->facts();
  const BackingFacts third_facts = test_case.third_backing->facts();
  const BackingFacts output_facts = test_case.output_backing->facts();
  std::vector<std::uint64_t> observed(test_case.expected.size());
  const bool first_output =
      test_case.output_backing->observe(
          std::as_writable_bytes(std::span{observed})) &&
      std::equal(observed.begin(), observed.end(), test_case.expected.begin());
  const bool route_evidence =
      test_case.state != nullptr &&
      test_case.state->geometry.route == detail::VirtualRoute::GraphPointwise &&
      test_case.state->device_vsm_product_cache == nullptr &&
      first_residency.window_handoff_count == 0u &&
      first_residency.window_batch_count == 0u &&
      first_residency.window_queue_call_count == 0u &&
      first_residency.prefetch_count == 0u;
  const std::uint64_t capacity =
      test_case.pipeline.plan().residency.frame_capacity;
  const std::uint64_t batches =
      capacity == 0u ? 0u : (GraphPageCount + capacity - 1u) / capacity;
  const bool graph_replay =
      backend != Backend::Vulkan ||
      (test_case.state != nullptr &&
       check_graph_replay(*test_case.state, capacity, batches));
  const bool valid =
      first_run && route_evidence && first_output && capacity == 2u &&
      graph_replay && first_residency.epoch_count == batches * StageCount &&
      first_residency.page_in_count == InputCount * GraphPageCount &&
      first_residency.backing_read_bytes ==
          sizeof(test_case.first_values) + sizeof(test_case.second_values) +
              sizeof(test_case.third_values) &&
      first_residency.page_out_count == GraphPageCount &&
      first_residency.backing_write_bytes == sizeof(test_case.expected) &&
      first_facts.read_bytes == sizeof(test_case.first_values) &&
      second_facts.read_bytes == sizeof(test_case.second_values) &&
      third_facts.read_bytes == sizeof(test_case.third_values) &&
      output_facts.write_count == GraphPageCount &&
      output_facts.write_bytes == sizeof(test_case.expected) &&
      detail::VirtualBackingAccess::version(*test_case.output_backing) ==
          initial_version + 1u &&
      test_case.second_backing->both_callbacks_started_before_release() &&
      test_case.second_backing->reversed() &&
      !test_case.second_backing->premature_reuse();
  if (valid) {
    return true;
  }

  std::fprintf(
      stderr,
      "virtual GraphPointwise Host wavefront backend=%u reason=%.*s "
      "route=%u cache=%u observed=%u k=%llu epochs=%llu page_in=%llu "
      "page_out=%llu reads=%llu/%llu/%llu writes=%llu reversed=%u "
      "premature=%u\n",
      static_cast<unsigned>(backend),
      static_cast<int>(first_run.error().size()), first_run.error().data(),
      test_case.state == nullptr
          ? 255u
          : static_cast<unsigned>(test_case.state->geometry.route),
      test_case.state != nullptr &&
              test_case.state->device_vsm_product_cache != nullptr
          ? 1u
          : 0u,
      first_output ? 1u : 0u, static_cast<unsigned long long>(capacity),
      static_cast<unsigned long long>(first_residency.epoch_count),
      static_cast<unsigned long long>(first_residency.page_in_count),
      static_cast<unsigned long long>(first_residency.page_out_count),
      static_cast<unsigned long long>(first_facts.read_bytes),
      static_cast<unsigned long long>(second_facts.read_bytes),
      static_cast<unsigned long long>(third_facts.read_bytes),
      static_cast<unsigned long long>(output_facts.write_count),
      test_case.second_backing->reversed() ? 1u : 0u,
      test_case.second_backing->premature_reuse() ? 1u : 0u);
  std::fprintf(
      stderr, " graph-submit failed=%u reason=%.*s owner=%u owner_ok=%u\n",
      route_observation.residency_submit_failed ? 1u : 0u,
      static_cast<int>(route_observation.residency_submit_reason_length),
      route_observation.residency_submit_reason.data(),
      route_observation.residency_submit_owner_valid ? 1u : 0u,
      route_observation.residency_submit_owner.ok ? 1u : 0u);
  if (test_case.state != nullptr && test_case.state->failure_log.has()) {
    const auto &failure = test_case.state->failure_log.first();
    std::fprintf(stderr,
                 " failure-log phase=%u check=%u reason=%u stage=%u batch=%llu "
                 "epoch=%llu token=%llu generation=%llu\n",
                 static_cast<unsigned>(failure.phase),
                 static_cast<unsigned>(failure.check),
                 static_cast<unsigned>(failure.reason),
                 static_cast<unsigned>(failure.stage),
                 static_cast<unsigned long long>(failure.batch),
                 static_cast<unsigned long long>(failure.epoch),
                 static_cast<unsigned long long>(failure.token),
                 static_cast<unsigned long long>(failure.generation));
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  if (test_case.state != nullptr) {
    for (std::size_t pipeline_index = 0u;
         pipeline_index < test_case.state->graph_pipelines.size();
         ++pipeline_index) {
      const auto &graph_pipeline =
          test_case.state->graph_pipelines[pipeline_index];
      rund::node::accel::detail::VulkanResidencyGraphGeneratedDiagnostics
          graph_diagnostics{};
      if (graph_pipeline == nullptr ||
          !rund::node::accel::detail::InspectVulkanGraphGeneratedDiagnostics(
              graph_pipeline->prepared, graph_diagnostics)) {
        continue;
      }
      std::fprintf(
          stderr,
          " graph-generated index=%llu phase=%u reason=%s generation=%llu "
          "admission=%u ready=%u recorded=%u submitted=%u terminal=%u "
          "accepted=%u known=%u unknown=%u replay=%u overflow=%u "
          "failed=%u quarantine=%u\n",
          static_cast<unsigned long long>(pipeline_index),
          static_cast<unsigned>(graph_diagnostics.phase),
          graph_diagnostics.first_reason == nullptr
              ? "null"
              : graph_diagnostics.first_reason,
          static_cast<unsigned long long>(graph_diagnostics.generation),
          graph_diagnostics.admission_count, graph_diagnostics.ready_count,
          graph_diagnostics.recorded_count, graph_diagnostics.submitted_count,
          graph_diagnostics.terminal_count, graph_diagnostics.accepted_count,
          graph_diagnostics.known_failure_count,
          graph_diagnostics.unknown_count, graph_diagnostics.replay ? 1u : 0u,
          graph_diagnostics.generation_overflow ? 1u : 0u,
          graph_diagnostics.failed ? 1u : 0u,
          graph_diagnostics.quarantined ? 1u : 0u);
    }
  }
#endif
  return false;
}

} // namespace rund_node_test_virtual::product::graph_wavefront_host
