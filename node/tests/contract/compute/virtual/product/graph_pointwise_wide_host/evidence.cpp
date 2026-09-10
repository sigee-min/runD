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
#include <span>

namespace rund_node_test_virtual::product::graph_pointwise_wide_host {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] bool
graph_generated_stage(const rund::compute::detail::PipelineState &pipeline,
                      const std::uint32_t data_bindings) noexcept {
  using namespace rund::node::accel::detail;
  VulkanResidencyAdmissionSnapshot snapshot{};
  if (!InspectVulkanResidencyAdmission(pipeline.prepared, snapshot) ||
      !snapshot.final_selected ||
      snapshot.final_candidate !=
          VulkanResidencyAdmissionCandidate::GraphStageGeneratedIndirect ||
      snapshot.selected_mode !=
          VulkanResidencyMode::GraphStageGeneratedIndirect) {
    return false;
  }
  const auto count = std::min<std::size_t>(snapshot.candidate_count,
                                           snapshot.candidates.size());
  for (std::size_t index = 0u; index < count; ++index) {
    const auto &candidate = snapshot.candidates[index];
    if (candidate.candidate ==
            VulkanResidencyAdmissionCandidate::GraphStageGeneratedIndirect &&
        candidate.reason == VulkanResidencyAdmissionReason::PredicateAccepted) {
      return candidate.entry_count == 2u && candidate.active_step_count == 2u &&
             candidate.data_binding_count == data_bindings &&
             candidate.control_binding_count == 1u;
    }
  }
  return false;
}

[[nodiscard]] bool graph_generated_wide(const Case &test_case) noexcept {
  using namespace rund::compute::detail::residency;
  if (test_case.state == nullptr ||
      test_case.state->graph_pipelines.size() != StageCount * Pool::BankCount) {
    return false;
  }
  for (std::size_t index = 0u; index < test_case.state->graph_pipelines.size();
       ++index) {
    const auto &pipeline = test_case.state->graph_pipelines[index];
    if (pipeline == nullptr ||
        !graph_generated_stage(*pipeline,
                               index / Pool::BankCount == 0u ? 8u : 2u)) {
      return false;
    }
  }
  return true;
}
#else
[[nodiscard]] constexpr bool graph_generated_wide(const Case &) noexcept {
  return true;
}
#endif

void report_route_evidence(
    const Case &test_case, const rund::compute::Backend backend,
    const rund_node_test_virtual::product::ProductRouteObservation
        &observation) noexcept {
  std::fprintf(stderr,
               "Graph wide route backend=%u demand=%u kind=%u completed=%u "
               "owners=0x%08x owner_count=%u conflicts=%u "
               "pipeline_called=%u pipeline_accepted=%u stream_called=%u "
               "stream_accepted=%u schedule_called=%u schedule_accepted=%u "
               "window_called=%u window_accepted=%u\n",
               static_cast<unsigned>(backend),
               static_cast<unsigned>(observation.demand),
               static_cast<unsigned>(observation.kind),
               observation.completed ? 1u : 0u, observation.accepted_owner_mask,
               observation.accepted_owner_count, observation.conflict_count,
               observation.submit_residency_pipeline_called ? 1u : 0u,
               observation.submit_residency_pipeline_accepted ? 1u : 0u,
               observation.submit_residency_stream_window_called ? 1u : 0u,
               observation.submit_residency_stream_window_accepted ? 1u : 0u,
               observation.submit_residency_schedule_called ? 1u : 0u,
               observation.submit_residency_schedule_accepted ? 1u : 0u,
               observation.window_submit_called ? 1u : 0u,
               observation.window_submit_accepted ? 1u : 0u);
  std::fprintf(
      stderr,
      "Graph wide submit failed=%u reason=%.*s owner_valid=%u owner_ok=%u\n",
      observation.residency_submit_failed ? 1u : 0u,
      static_cast<int>(observation.residency_submit_reason_length),
      observation.residency_submit_reason.data(),
      observation.residency_submit_owner_valid ? 1u : 0u,
      observation.residency_submit_owner.ok ? 1u : 0u);

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using namespace rund::node::accel::detail;
  if (backend != rund::compute::Backend::Vulkan || test_case.state == nullptr) {
    return;
  }

  const auto captured_owner = observation.residency_submit_owner_valid
                                  ? observation.residency_submit_owner.owner
                                  : nullptr;
  const auto report_owner = [&](const std::size_t index, const char *const slot,
                                const PreparedKernelPipeline &prepared) {
    const bool present = prepared.owner != nullptr;
    const bool match = present && captured_owner != nullptr &&
                       prepared.owner == captured_owner;
    std::fprintf(stderr,
                 "Graph wide owner index=%zu slot=%s present=%u match=%u\n",
                 index, slot, present ? 1u : 0u, match ? 1u : 0u);
    if (!present) {
      return;
    }

    VulkanResidencyAdmissionSnapshot admission{};
    VulkanResidencyStatus native{};
    VulkanResidencyGraphGeneratedDiagnostics diagnostics{};
    const bool admission_ok =
        InspectVulkanResidencyAdmission(prepared, admission);
    const bool native_ok = InspectVulkanResidencyStatus(prepared, native);
    const bool diagnostics_ok =
        InspectVulkanGraphGeneratedDiagnostics(prepared, diagnostics);
    std::fprintf(
        stderr,
        "Graph wide selection index=%zu slot=%s available=%u final=%u "
        "candidate=%u mode=%u generation=%llu candidates=%u\n",
        index, slot, admission_ok ? 1u : 0u,
        admission_ok && admission.final_selected ? 1u : 0u,
        admission_ok ? static_cast<unsigned>(admission.final_candidate) : 255u,
        admission_ok ? static_cast<unsigned>(admission.selected_mode) : 255u,
        static_cast<unsigned long long>(admission_ok ? admission.generation_tag
                                                     : 0u),
        admission_ok ? admission.candidate_count : 0u);
    std::fprintf(
        stderr,
        "Graph wide native index=%zu slot=%s available=%u valid=%u present=%u "
        "ready=%u bounded=%u mode=%u locals=%u commands=%u steps=%u "
        "prefix=%u fence=%u suffix=%u args=%u mapped=%u quarantine=%u "
        "adapter_quarantine=%u reason=%s\n",
        index, slot, native_ok ? 1u : 0u, native.valid ? 1u : 0u,
        native.present ? 1u : 0u, native.ready ? 1u : 0u,
        native.bounded ? 1u : 0u, static_cast<unsigned>(native.mode),
        native.local_count, native.command_count, native.step_buffer_count,
        native.prefix_buffer ? 1u : 0u, native.prefix_fence ? 1u : 0u,
        native.suffix_buffer ? 1u : 0u, native.arguments_buffer ? 1u : 0u,
        native.arguments_mapped ? 1u : 0u, native.quarantined ? 1u : 0u,
        native.adapter_quarantined ? 1u : 0u,
        native.readiness_reason == nullptr ? "<null>"
                                           : native.readiness_reason);
    std::fprintf(
        stderr,
        "Graph wide generated index=%zu slot=%s available=%u phase=%u "
        "generation=%llu record=%llu submit_generation=%llu "
        "admission=%u ready=%u recorded=%u submitted=%u terminal=%u "
        "accepted=%u known=%u unknown=%u seq=%llu/%llu totals=%llu/%llu/%llu "
        "known_total=%llu unknown_total=%llu failed=%u quarantine=%u\n",
        index, slot, diagnostics_ok ? 1u : 0u,
        diagnostics_ok ? static_cast<unsigned>(diagnostics.phase) : 255u,
        static_cast<unsigned long long>(diagnostics_ok ? diagnostics.generation
                                                       : 0u),
        static_cast<unsigned long long>(
            diagnostics_ok ? diagnostics.record_generation : 0u),
        static_cast<unsigned long long>(
            diagnostics_ok ? diagnostics.submission_generation : 0u),
        diagnostics_ok ? diagnostics.admission_count : 0u,
        diagnostics_ok ? diagnostics.ready_count : 0u,
        diagnostics_ok ? diagnostics.recorded_count : 0u,
        diagnostics_ok ? diagnostics.submitted_count : 0u,
        diagnostics_ok ? diagnostics.terminal_count : 0u,
        diagnostics_ok ? diagnostics.accepted_count : 0u,
        diagnostics_ok ? diagnostics.known_failure_count : 0u,
        diagnostics_ok ? diagnostics.unknown_count : 0u,
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
            diagnostics_ok ? diagnostics.unknown_total : 0u),
        diagnostics_ok && diagnostics.failed ? 1u : 0u,
        diagnostics_ok && diagnostics.quarantined ? 1u : 0u);
    if (!admission_ok) {
      return;
    }
    const auto count = std::min<std::size_t>(admission.candidate_count,
                                             admission.candidates.size());
    for (std::size_t candidate_index = 0u; candidate_index < count;
         ++candidate_index) {
      const auto &candidate = admission.candidates[candidate_index];
      std::fprintf(
          stderr,
          "Graph wide candidate index=%zu slot=%s ordinal=%zu candidate=%u "
          "reason=%u failure_predicate=%u failure_ordinal=%u "
          "failure_reason=%s entry=%u local=%u predicate_ordinal=%u "
          "entries=%u active_steps=%u data_bindings=%u control_bindings=%u "
          "generation=%llu\n",
          index, slot, candidate_index,
          static_cast<unsigned>(candidate.candidate),
          static_cast<unsigned>(candidate.reason),
          static_cast<unsigned>(candidate.first_failure_predicate),
          candidate.first_failure_ordinal,
          candidate.first_failure_reason == nullptr
              ? "<null>"
              : candidate.first_failure_reason,
          candidate.entry_ordinal, candidate.local_ordinal,
          candidate.predicate_ordinal, candidate.entry_count,
          candidate.active_step_count, candidate.data_binding_count,
          candidate.control_binding_count,
          static_cast<unsigned long long>(candidate.generation_tag));
    }
  };

  for (std::size_t index = 0u; index < test_case.state->graph_pipelines.size();
       ++index) {
    const auto &graph_pipeline = test_case.state->graph_pipelines[index];
    if (graph_pipeline == nullptr) {
      std::fprintf(
          stderr, "Graph wide owner index=%zu slot=missing present=0 match=0\n",
          index);
      continue;
    }
    report_owner(index, "primary", graph_pipeline->prepared);
    report_owner(index, "alternate", graph_pipeline->alternate_prepared);
  }
#else
  (void)test_case;
  (void)backend;
  (void)observation;
#endif
}

bool validate_case(Case &test_case, const rund::compute::Backend backend,
                   const rund::compute::Status &status,
                   const std::uint64_t initial_version) noexcept {
  using namespace rund::compute;
  constexpr std::size_t ElementCount = PageCount * FrameElements - TailElements;
  constexpr std::size_t LogicalBytes = ElementCount * sizeof(std::uint64_t);
  const Stats stats = test_case.pipeline.stats();
  const ResidencyStats residency = stats.pipeline.residency;
  bool inputs_valid = true;
  for (const auto &backing : test_case.input_backings) {
    const BackingFacts facts = backing->facts();
    inputs_valid = inputs_valid && facts.read_count == PageCount &&
                   facts.read_bytes == LogicalBytes;
  }
  std::vector<std::uint64_t> observed(ElementCount);
  const BackingFacts output = test_case.output_backing->facts();
  const std::uint64_t batches = (PageCount + 1u) / 2u;
  const bool valid =
      status && test_case.state != nullptr &&
      test_case.state->device_vsm_product_cache == nullptr &&
      test_case.state->graph_pipelines.size() ==
          StageCount * rund::compute::detail::residency::Pool::BankCount &&
      test_case.pipeline.plan().residency.frame_capacity == 2u &&
      (backend != Backend::Vulkan || graph_generated_wide(test_case)) &&
      residency.epoch_count == batches * StageCount &&
      residency.page_in_count == InputCount * PageCount &&
      residency.backing_read_bytes == InputCount * LogicalBytes &&
      residency.page_out_count == PageCount &&
      residency.backing_write_bytes == LogicalBytes && inputs_valid &&
      output.write_count == PageCount && output.write_bytes == LogicalBytes &&
      detail::VirtualBackingAccess::version(*test_case.output_backing) ==
          initial_version + 1u &&
      detail::VirtualBackingAccess::recovery_bytes(*test_case.output_backing) ==
          0u &&
      test_case.output_backing->observe(
          std::as_writable_bytes(std::span{observed})) &&
      observed == test_case.expected &&
      test_case.output_backing->tail_poisoned();
  if (!valid) {
    std::fprintf(stderr,
                 "Graph wide Host backend=%u reason=%.*s epoch=%llu page=%llu/"
                 "%llu bytes=%llu/%llu graphs=%zu vsm=%u\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(status.error().size()), status.error().data(),
                 static_cast<unsigned long long>(residency.epoch_count),
                 static_cast<unsigned long long>(residency.page_in_count),
                 static_cast<unsigned long long>(residency.page_out_count),
                 static_cast<unsigned long long>(residency.backing_read_bytes),
                 static_cast<unsigned long long>(residency.backing_write_bytes),
                 test_case.state == nullptr
                     ? 0u
                     : test_case.state->graph_pipelines.size(),
                 static_cast<unsigned>(
                     test_case.state != nullptr &&
                     test_case.state->device_vsm_product_cache != nullptr));
  }
  return valid;
}

} // namespace rund_node_test_virtual::product::graph_pointwise_wide_host
