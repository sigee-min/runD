#include "local.hpp"
#include "plan.hpp"
#include "mode.hpp"
#include "graph_direct.hpp"

#include "generated_indirect/internal.hpp"
#include "generated_indirect/map.hpp"

#include "../prepare/record.hpp"

#include "../../../../kernel/recurrence.hpp"
#include "../../../command/resources.hpp"
#include "../../../map/api.hpp"
#include "../../../map/local.hpp"
#include "../../ops/table.hpp"

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

void RecordResidencyAdmissionCandidate(
    VulkanPipeline &pipeline, const std::size_t index,
    const VulkanResidencyAdmissionReason reason,
    const std::uint32_t entry_ordinal = VulkanResidencyAdmissionNoOrdinal,
    const VulkanResidencyGraphGeneratedPredicate first_failure_predicate =
        VulkanResidencyGraphGeneratedPredicate::None,
    const std::uint32_t first_failure_ordinal =
        VulkanResidencyAdmissionNoOrdinal,
    const char *const first_failure_reason = "not_evaluated") noexcept {
  auto &candidate = pipeline.residency_admission.candidates[index];
  candidate.reason = reason;
  candidate.entry_ordinal = entry_ordinal;
  candidate.local_ordinal = entry_ordinal;
  candidate.first_failure_predicate = first_failure_predicate;
  candidate.first_failure_ordinal = first_failure_ordinal;
  candidate.first_failure_reason = first_failure_reason;
  if (pipeline.record == nullptr) {
    return;
  }
  candidate.entry_count = static_cast<std::uint32_t>(
      std::min<std::size_t>(pipeline.record->entries.size(),
                            std::numeric_limits<std::uint32_t>::max()));
  candidate.active_step_count = pipeline.record->status.active_step_count;
  if (entry_ordinal >= pipeline.record->entries.size()) {
    return;
  }
  const auto &entry = pipeline.record->entries[entry_ordinal];
  const auto *const resources =
      entry.prepared == nullptr
          ? nullptr
          : static_cast<const VulkanKernelResources *>(entry.prepared.get());
  const auto *const kernel =
      resources == nullptr ? nullptr : resources->entry(0u);
  if (kernel == nullptr) {
    return;
  }
  const BackendRun *const run =
      resources->program == nullptr ? nullptr : resources->program->signature;
  const BoundStep *const bound =
      kernel->view != nullptr ? &kernel->view->step
                              : (run == nullptr || run->steps == nullptr ||
                                         entry_ordinal >= run->step_count
                                     ? nullptr
                                     : &run->steps[entry_ordinal]);
  if (bound != nullptr && bound->step != nullptr) {
    candidate.data_binding_count =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            bound->step->artifact.metadata.binding_accesses.size(),
            std::numeric_limits<std::uint32_t>::max()));
  }
  const auto *const map = kernel->resource == nullptr
                              ? nullptr
                              : static_cast<const VulkanMapEncodeResources *>(
                                    kernel->resource.get());
  if (map != nullptr) {
    candidate.control_binding_count =
        static_cast<std::uint32_t>(map->control.has_count()) +
        static_cast<std::uint32_t>(map->control.has_predicate());
    candidate.predicate_ordinal = map->control.has_predicate()
                                      ? map->control.predicate_binding
                                      : VulkanResidencyAdmissionNoOrdinal;
  }
}

void RecordAllResidencyAdmissionCandidates(
    VulkanPipeline &pipeline,
    const VulkanResidencyAdmissionReason reason) noexcept {
  for (std::size_t index = 0u; index < 5u; ++index) {
    RecordResidencyAdmissionCandidate(pipeline, index, reason);
  }
}

void RecordGraphGeneratedProof(
    VulkanPipeline &pipeline,
    const VulkanResidencyGraphStageGeneratedProof &proof) noexcept {
  auto &candidate = pipeline.residency_admission.candidates[1u];
  candidate.data_binding_count = proof.binding_count;
  candidate.control_binding_count =
      static_cast<std::uint32_t>(proof.has_count) +
      static_cast<std::uint32_t>(proof.has_predicate);
}

} // namespace

[[nodiscard]] bool SelectionEntry(const VulkanPipeline &pipeline,
                                  const std::size_t index,
                                  VulkanKernelResources *&resources) noexcept {
  resources = nullptr;
  if (pipeline.record == nullptr || index >= pipeline.record->entries.size()) {
    return false;
  }
  const VulkanPipelineRecordEntry &entry = pipeline.record->entries[index];
  if (entry.prepared == nullptr || entry.window.has_value() ||
      entry.transducer != NoTileTransducer || entry.template_index != index ||
      entry.occurrence_index != index) {
    return false;
  }
  resources = static_cast<VulkanKernelResources *>(entry.prepared.get());
  return resources != nullptr && resources->size() != 0u &&
         resources->dispatch_count != 0u;
}

[[nodiscard]] bool StepControlCount(const VulkanPipeline &pipeline,
                                    const std::size_t index,
                                    std::uint64_t &count) noexcept {
  count = 0u;
  if (pipeline.record == nullptr || index >= pipeline.record->template_count) {
    return false;
  }
  const VulkanPipelineRecordRecipe &recipe = *pipeline.record;
  const PreparedProgramStatusSlice status = recipe.status_ranges[index];
  const PreparedProgramStatusSlice telemetry = recipe.telemetry_ranges[index];
  const std::size_t status_end =
      static_cast<std::size_t>(status.first) + status.count;
  const std::size_t telemetry_end =
      static_cast<std::size_t>(telemetry.first) + telemetry.count;
  if (status_end > recipe.status_steps.size() ||
      telemetry_end > recipe.telemetry_steps.size()) {
    return false;
  }
  for (std::size_t current = status.first; current < status_end; ++current) {
    const std::uint64_t sources = recipe.status_steps[current].count;
    if (sources > (std::numeric_limits<std::uint64_t>::max() - count) / 2u) {
      return false;
    }
    count += 2u * sources;
  }
  for (std::size_t current = telemetry.first; current < telemetry_end;
       ++current) {
    const std::uint64_t sources = recipe.telemetry_steps[current].count;
    if (sources > std::numeric_limits<std::uint64_t>::max() - count) {
      return false;
    }
    count += sources;
  }
  return true;
}

[[nodiscard]] rund::AccelCheck
BuildResidencyPlan(VulkanPipeline &pipeline, ResidencyPlan &plan) noexcept {
  ResetResidencyAdmissionSnapshot(pipeline);
  plan = ResidencyPlan{};
  if (!ValidVulkanPipeline(&pipeline) || pipeline.record == nullptr ||
      pipeline.profile != nullptr || pipeline.record->recurrence ||
      pipeline.record->entries.empty() || !pipeline.transducers.empty() ||
      !pipeline.window.routes.empty() || !pipeline.window.gates.empty() ||
      !pipeline.publish.routes.empty() ||
      pipeline.record->entries.size() != pipeline.record->barriers.size() ||
      pipeline.record->entries.size() != pipeline.record->template_count ||
      pipeline.record->entries.size() !=
          pipeline.record->status.active_step_count ||
      pipeline.record->status.active_step_count !=
          pipeline.record->status.declared_step_count ||
      pipeline.record->entries.size() > PreparedPipelineStepCapacity) {
    RecordAllResidencyAdmissionCandidates(
        pipeline, VulkanResidencyAdmissionReason::InvalidPipeline);
    return {false, "accel_vulkan_residency_not_applicable"};
  }
  if (BuildGraphDirectProof(pipeline, plan.graph_direct)) {
    RecordResidencyAdmissionCandidate(
        pipeline, 0u, VulkanResidencyAdmissionReason::PredicateAccepted);
    RecordResidencyAdmissionCandidate(
        pipeline, 1u, VulkanResidencyAdmissionReason::PredicateRejected);
    RecordResidencyAdmissionCandidate(
        pipeline, 2u, VulkanResidencyAdmissionReason::PredicateRejected);
    RecordResidencyAdmissionCandidate(
        pipeline, 3u, VulkanResidencyAdmissionReason::PredicateRejected);
    RecordResidencyAdmissionCandidate(
        pipeline, 4u, VulkanResidencyAdmissionReason::PredicateRejected);
    plan.candidate = VulkanResidencyAdmissionCandidate::GraphDirect;
    plan.bounded = true;
    return {true, "ok"};
  }
  RecordResidencyAdmissionCandidate(
      pipeline, 0u, VulkanResidencyAdmissionReason::PredicateRejected);
  const vulkan_generated_indirect_detail::GraphAdmission graph =
      vulkan_generated_indirect_detail::admit_graph(pipeline);
  plan.graph_generated = graph.graph;
  plan.graph_sequence = graph.sequence;
  if (graph.candidate.has_value() &&
      *graph.candidate ==
          VulkanResidencyAdmissionCandidate::GraphStageGeneratedIndirect) {
    RecordResidencyAdmissionCandidate(
        pipeline, 1u, VulkanResidencyAdmissionReason::PredicateAccepted);
    RecordResidencyAdmissionCandidate(
        pipeline, 2u, VulkanResidencyAdmissionReason::PredicateRejected);
    RecordResidencyAdmissionCandidate(
        pipeline, 3u, VulkanResidencyAdmissionReason::PredicateRejected);
    RecordResidencyAdmissionCandidate(
        pipeline, 4u, VulkanResidencyAdmissionReason::PredicateRejected);
    RecordGraphGeneratedProof(pipeline, plan.graph_generated);
    plan.candidate =
        VulkanResidencyAdmissionCandidate::GraphStageGeneratedIndirect;
    plan.bounded = true;
    return {true, "ok"};
  }
  RecordResidencyAdmissionCandidate(
      pipeline, 1u, VulkanResidencyAdmissionReason::PredicateRejected,
      VulkanResidencyAdmissionNoOrdinal,
      plan.graph_generated.first_failure_predicate,
      plan.graph_generated.first_failure_ordinal,
      plan.graph_generated.first_failure_reason);
  if (graph.candidate.has_value() &&
      *graph.candidate ==
          VulkanResidencyAdmissionCandidate::GraphStageSequence) {
    RecordResidencyAdmissionCandidate(
        pipeline, 2u, VulkanResidencyAdmissionReason::PredicateAccepted);
    RecordResidencyAdmissionCandidate(
        pipeline, 3u, VulkanResidencyAdmissionReason::PredicateRejected);
    RecordResidencyAdmissionCandidate(
        pipeline, 4u, VulkanResidencyAdmissionReason::PredicateRejected);
    plan.candidate = VulkanResidencyAdmissionCandidate::GraphStageSequence;
    plan.bounded = true;
    return {true, "ok"};
  }
  RecordResidencyAdmissionCandidate(
      pipeline, 2u, VulkanResidencyAdmissionReason::PredicateRejected,
      VulkanResidencyAdmissionNoOrdinal,
      plan.graph_sequence.first_failure_predicate,
      plan.graph_sequence.first_failure_ordinal,
      plan.graph_sequence.first_failure_reason);
  const bool generated = vulkan_generated_indirect_detail::eligible(pipeline);
  RecordResidencyAdmissionCandidate(
      pipeline, 3u,
      generated ? VulkanResidencyAdmissionReason::PredicateAccepted
                : VulkanResidencyAdmissionReason::PredicateRejected);
  bool direct_ok = true;
  std::uint32_t generic_failure = VulkanResidencyAdmissionNoOrdinal;
  for (std::size_t index = 0u; index < pipeline.record->entries.size();
       ++index) {
    VulkanKernelResources *resources = nullptr;
    std::uint64_t control_count = 0u;
    std::uint64_t direct = 0u;
    std::uint64_t indirect = 0u;
    if (pipeline.record->status.declared_steps[index] != index ||
        !SelectionEntry(pipeline, index, resources) ||
        !StepControlCount(pipeline, index, control_count) ||
        !DescribeVulkanRouteDispatches(*resources, direct, indirect).ok ||
        (direct == 0u && indirect == 0u) || (indirect != 0u && !generated)) {
      direct_ok = false;
      generic_failure = static_cast<std::uint32_t>(index);
      break;
    }
    if (indirect != 0u) {
      direct_ok = false;
      generic_failure = static_cast<std::uint32_t>(index);
      break;
    }
  }
  RecordResidencyAdmissionCandidate(
      pipeline, 4u,
      direct_ok ? VulkanResidencyAdmissionReason::PredicateAccepted
                : VulkanResidencyAdmissionReason::PredicateRejected,
      generic_failure);
  if (!direct_ok && !generated) {
    return {false, "accel_vulkan_residency_not_applicable"};
  }
  plan.candidate = generated
                       ? VulkanResidencyAdmissionCandidate::GeneratedIndirect
                       : VulkanResidencyAdmissionCandidate::Direct;
  plan.bounded = direct_ok;
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
