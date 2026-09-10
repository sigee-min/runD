#include "local.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace rund_node_test_virtual::product::reduce::sequence_detail {

void PrintTiledGraphEvidence(
    const rund::compute::Backend backend,
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState> &state,
    const ProductRouteObservation &observation) {
  std::fprintf(stderr,
               " graph-submit backend=%u failed=%u reason=%.*s owner=%u "
               "owner_ok=%u\n",
               static_cast<unsigned>(backend),
               observation.residency_submit_failed ? 1u : 0u,
               static_cast<int>(observation.residency_submit_reason_length),
               observation.residency_submit_reason.data(),
               observation.residency_submit_owner_valid ? 1u : 0u,
               observation.residency_submit_owner.ok ? 1u : 0u);
  if (backend != rund::compute::Backend::Vulkan || state == nullptr) {
    return;
  }
  for (std::size_t index = 0u; index < state->graph_pipelines.size(); ++index) {
    const auto &graph_pipeline = state->graph_pipelines[index];
    if (graph_pipeline == nullptr) {
      continue;
    }
    using namespace rund::node::accel::detail;
    const auto &primary = graph_pipeline->prepared;
    const auto &alternate = graph_pipeline->alternate_prepared;
    const bool captured_owner =
        observation.residency_submit_owner_valid &&
        observation.residency_submit_owner.owner != nullptr;
    const bool primary_match =
        captured_owner &&
        primary.owner == observation.residency_submit_owner.owner;
    const bool alternate_match =
        captured_owner &&
        alternate.owner == observation.residency_submit_owner.owner;
    const bool owner_match = primary_match || alternate_match;
    const auto *matched = primary_match     ? &primary
                          : alternate_match ? &alternate
                                            : nullptr;
    VulkanResidencyStatus native{};
    VulkanResidencyAdmissionSnapshot admission{};
    VulkanResidencyGraphGeneratedDiagnostics diagnostics{};
    const bool native_ok =
        matched != nullptr && InspectVulkanResidencyStatus(*matched, native);
    const bool admission_ok =
        matched != nullptr &&
        InspectVulkanResidencyAdmission(*matched, admission);
    const bool diagnostics_ok =
        matched != nullptr &&
        InspectVulkanGraphGeneratedDiagnostics(*matched, diagnostics);
    const std::size_t bank = index % 2u;
    const std::size_t stage = index / 2u;
    std::fprintf(stderr,
                 " graph-owner index=%zu stage=%zu bank=%zu slot=%zu parity=%u "
                 "match=%u primary_match=%u alternate_match=%u\n",
                 index, stage, bank, index % 2u,
                 static_cast<unsigned>(graph_pipeline->attempt.parity),
                 owner_match ? 1u : 0u, primary_match ? 1u : 0u,
                 alternate_match ? 1u : 0u);
    if (!owner_match) {
      std::fprintf(stderr,
                   " graph-owner-no-match index=%zu primary_present=%u "
                   "alternate_present=%u\n",
                   index, primary.owner != nullptr ? 1u : 0u,
                   alternate.owner != nullptr ? 1u : 0u);
      continue;
    }
    std::fprintf(
        stderr,
        " graph-native-status index=%zu available=%u valid=%u present=%u "
        "ready=%u bounded=%u mode=%u quarantine=%u adapter_quarantine=%u "
        "locals=%u commands=%u step_buffers=%u prefix_buffer=%u "
        "prefix_fence=%u suffix_buffer=%u arguments_buffer=%u "
        "arguments_mapped=%u reason=%s\n",
        index, native_ok ? 1u : 0u, native.valid ? 1u : 0u,
        native.present ? 1u : 0u, native.ready ? 1u : 0u,
        native.bounded ? 1u : 0u, static_cast<unsigned>(native.mode),
        native.quarantined ? 1u : 0u, native.adapter_quarantined ? 1u : 0u,
        native.local_count, native.command_count, native.step_buffer_count,
        native.prefix_buffer ? 1u : 0u, native.prefix_fence ? 1u : 0u,
        native.suffix_buffer ? 1u : 0u, native.arguments_buffer ? 1u : 0u,
        native.arguments_mapped ? 1u : 0u,
        native.readiness_reason == nullptr ? "<null>"
                                           : native.readiness_reason);
    std::fprintf(
        stderr,
        " graph-selection index=%zu admission=%u final_selected=%u "
        "final_candidate=%u selected_mode=%u generation=%llu "
        "diag=%u phase=%u generation=%llu record_generation=%llu "
        "submission_generation=%llu seq=%llu/%llu totals=%llu/%llu/%llu "
        "known=%llu unknown=%llu\n",
        index, admission_ok ? 1u : 0u,
        admission_ok && admission.final_selected ? 1u : 0u,
        admission_ok ? static_cast<unsigned>(admission.final_candidate) : 255u,
        admission_ok ? static_cast<unsigned>(admission.selected_mode) : 255u,
        static_cast<unsigned long long>(admission_ok ? admission.generation_tag
                                                     : 0u),
        diagnostics_ok ? 1u : 0u,
        diagnostics_ok ? static_cast<unsigned>(diagnostics.phase) : 255u,
        static_cast<unsigned long long>(diagnostics_ok ? diagnostics.generation
                                                       : 0u),
        static_cast<unsigned long long>(
            diagnostics_ok ? diagnostics.record_generation : 0u),
        static_cast<unsigned long long>(
            diagnostics_ok ? diagnostics.submission_generation : 0u),
        static_cast<unsigned long long>(diagnostics_ok ? diagnostics.submit_seq
                                                       : 0u),
        static_cast<unsigned long long>(diagnostics_ok ? diagnostics.done_seq
                                                       : 0u),
        static_cast<unsigned long long>(
            diagnostics_ok ? diagnostics.submit_total : 0u),
        static_cast<unsigned long long>(diagnostics_ok ? diagnostics.done_total
                                                       : 0u),
        static_cast<unsigned long long>(
            diagnostics_ok ? diagnostics.accept_total : 0u),
        static_cast<unsigned long long>(diagnostics_ok ? diagnostics.known_total
                                                       : 0u),
        static_cast<unsigned long long>(
            diagnostics_ok ? diagnostics.unknown_total : 0u));
    if (admission_ok) {
      const auto count = std::min<std::size_t>(admission.candidate_count,
                                               admission.candidates.size());
      for (std::size_t candidate = 0u; candidate < count; ++candidate) {
        const auto &row = admission.candidates[candidate];
        std::fprintf(
            stderr,
            " graph-candidate index=%zu ordinal=%zu candidate=%u reason=%u "
            "predicate=%u failure_ordinal=%u entry=%u local=%u\n",
            index, candidate, static_cast<unsigned>(row.candidate),
            static_cast<unsigned>(row.reason),
            static_cast<unsigned>(row.first_failure_predicate),
            row.first_failure_ordinal, row.entry_ordinal, row.local_ordinal);
      }
    }
  }
}

[[nodiscard]] bool InspectGraphSequence(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState> &state,
    const bool after_run) noexcept {
  if (state == nullptr || state->graph_pipelines.size() < 2u) {
    return false;
  }
  for (std::size_t bank = 0u; bank < 2u; ++bank) {
    const auto &graph_pipeline = state->graph_pipelines[bank];
    if (graph_pipeline == nullptr ||
        graph_pipeline->prepared.owner == nullptr) {
      return false;
    }
    using namespace rund::node::accel::detail;
    VulkanResidencyAdmissionSnapshot admission{};
    VulkanResidencyStatus native{};
    VulkanResidencyGraphGeneratedDiagnostics diagnostics{};
    if (!InspectVulkanResidencyAdmission(graph_pipeline->prepared, admission) ||
        !InspectVulkanResidencyStatus(graph_pipeline->prepared, native) ||
        !InspectVulkanGraphGeneratedDiagnostics(graph_pipeline->prepared,
                                                diagnostics)) {
      return false;
    }
    bool sequence_accepted = false;
    const std::size_t candidate_count = std::min<std::size_t>(
        admission.candidate_count, admission.candidates.size());
    for (std::size_t index = 0u; index < candidate_count; ++index) {
      const auto &candidate = admission.candidates[index];
      if (candidate.candidate ==
              VulkanResidencyAdmissionCandidate::GraphStageSequence &&
          candidate.reason ==
              VulkanResidencyAdmissionReason::PredicateAccepted) {
        sequence_accepted = true;
        break;
      }
    }
    if (!sequence_accepted || !admission.final_selected ||
        admission.final_candidate !=
            VulkanResidencyAdmissionCandidate::GraphStageSequence ||
        admission.selected_mode != VulkanResidencyMode::GraphStageSequence ||
        admission.generation_tag == 0u || !native.valid || !native.present ||
        !native.ready || !native.bounded ||
        native.mode != VulkanResidencyMode::GraphStageSequence ||
        native.local_count != 2u || native.command_count != 2u ||
        native.quarantined || native.adapter_quarantined ||
        native.readiness_reason == nullptr ||
        std::strcmp(native.readiness_reason, "ok") != 0 ||
        diagnostics.phase != VulkanResidencyGraphGeneratedPhase::Recorded ||
        diagnostics.generation == 0u || diagnostics.admission_count != 4u ||
        diagnostics.ready_count != 4u || diagnostics.recorded_count != 4u ||
        diagnostics.failed || diagnostics.quarantined) {
      return false;
    }
    if (!after_run) {
      if (diagnostics.submit_seq != 0u || diagnostics.done_seq != 0u ||
          diagnostics.submit_total != 0u || diagnostics.done_total != 0u ||
          diagnostics.accept_total != 0u || diagnostics.known_total != 0u ||
          diagnostics.unknown_total != 0u ||
          diagnostics.submitted_count != 0u ||
          diagnostics.terminal_count != 0u ||
          diagnostics.accepted_count != 0u ||
          diagnostics.known_failure_count != 0u ||
          diagnostics.unknown_count != 0u) {
        return false;
      }
      continue;
    }
    constexpr std::uint64_t TotalRuns = GraphWarmRuns + 1u;
    const std::uint64_t expected_seq = TotalRuns * (bank == 0u ? 2u : 1u);
    const std::uint64_t expected_total = TotalRuns * (bank == 0u ? 6u : 4u);
    if (diagnostics.submit_seq != expected_seq ||
        diagnostics.done_seq != expected_seq ||
        diagnostics.submit_total != expected_total ||
        diagnostics.done_total != expected_total ||
        diagnostics.accept_total != expected_total ||
        diagnostics.known_total != 0u || diagnostics.unknown_total != 0u ||
        diagnostics.submitted_count != 0u || diagnostics.terminal_count != 0u ||
        diagnostics.accepted_count != 0u ||
        diagnostics.known_failure_count != 0u ||
        diagnostics.unknown_count != 0u) {
      return false;
    }
  }
  return true;
}

} // namespace rund_node_test_virtual::product::reduce::sequence_detail

#endif
