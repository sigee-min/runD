#include "../../state/assembly.hpp"
#include "model.hpp"
#include "../publication.hpp"

#include "../../claim.hpp"
#include "../../local.hpp"

#include <cstdint>
#include <limits>
#include <utility>

namespace rund::compute::detail {

Status admit_publications(AdmissionDraft &draft) {
  PipelineState *const state = draft.state.get();
  const PipelineMemoryPlan &plan = draft.plan;
  auto &hash = draft.hash;
  std::size_t &output_count = draft.output_count;

  const auto exact_resource =
      [&](const PipelinePublicationViewPlan &view) -> PipelineResource * {
    const PipelinePublicationViewIdentity &identity = view.identity;
    if (identity.resource_ordinal >= state->resources.size() ||
        identity.resource_ordinal >= plan.resources.size()) {
      return nullptr;
    }
    PipelineResource &resource = state->resources[identity.resource_ordinal];
    return resource.buffer != nullptr && resource.type == view.type &&
                   resource.format == view.format &&
                   resource.buffer->type == view.type &&
                   resource.bytes == identity.backing_bytes &&
                   resource.buffer->bytes == identity.backing_bytes &&
                   identity.element_bytes != 0u &&
                   identity.stride_bytes != 0u &&
                   identity.offset_bytes <= identity.backing_bytes &&
                   (identity.count == 0u ||
                    (identity.element_bytes <=
                         identity.backing_bytes - identity.offset_bytes &&
                     identity.count - 1u <=
                         (identity.backing_bytes - identity.offset_bytes -
                          identity.element_bytes) /
                             identity.stride_bytes))
               ? &resource
               : nullptr;
  };

  state->publications.reserve(plan.publications.size());
  hash.number(plan.publications.size());
  for (const PipelinePublicationPlan &planned : plan.publications) {
    const auto *window = std::get_if<PipelineWindowPublicationPlan>(&planned);
    const auto *terminal =
        std::get_if<PipelineTerminalPublicationPlan>(&planned);
    const std::uint32_t planned_state =
        window != nullptr ? window->state : terminal->state;
    const std::uint32_t physical =
        window != nullptr ? window->output.value : terminal->output.value;
    if (planned_state >= state->windows.size() ||
        physical >= PipelineLeafCapacity ||
        physical > std::numeric_limits<std::uint16_t>::max()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const PipelineWindow &publication_window = state->windows[planned_state];
    const PipelineWindowControl &control = publication_window.control;
    const PipelinePublicationTargetPlan &target_plan =
        pipeline_publication_target(planned);
    const std::uint32_t target = target_plan.view.identity.resource_ordinal;
    if (target >= plan.resources.size() || target >= state->resources.size() ||
        !std::holds_alternative<PipelineExternalResourcePlan>(
            plan.resources[target].locator)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const Status target_device =
        validate_pipeline_resource_device(*state, state->resources[target]);
    if (!target_device) {
      return target_device;
    }
    if (exact_resource(target_plan.view) == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (window != nullptr) {
      if (exact_resource(window->source) == nullptr ||
          exact_resource(control.count) == nullptr || control.maximum == 0u ||
          control.tile == 0u || control.tile > control.maximum) {
        return Status::fail(Reason::PipelineInvalid);
      }
    } else {
      if (terminal == nullptr || control.final < PipelineWindow::first ||
          control.final > PipelineWindow::second ||
          control.final >= terminal->sources.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      for (const PipelinePublicationViewPlan &bank : terminal->sources) {
        if (exact_resource(bank) == nullptr) {
          return Status::fail(Reason::PipelineInvalid);
        }
      }
    }
    PipelineResource &target_resource = state->resources[target];
    if (!plan.resources[target].output ||
        target_resource.output != PipelineResource::no_output) {
      return Status::fail(target_resource.output != PipelineResource::no_output
                              ? Reason::BindingDuplicate
                              : Reason::PipelineInvalid);
    }
    target_resource.output = 0u;
    target_resource.terminal_publish = window == nullptr;
    ++output_count;
    state->publications.push_back(planned);
    if (!mix_pipeline_publication_public_identity(hash, planned, control)) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  for (std::size_t ordinal = 0u; ordinal < plan.resources.size(); ++ordinal) {
    const PipelineResolvedResourcePlan &planned = plan.resources[ordinal];
    const PipelineResource &admitted = state->resources[ordinal];
    if (planned.output != (admitted.output != PipelineResource::no_output) ||
        (planned.output &&
         planned.terminal_publish != admitted.terminal_publish)) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }


  return Status::success();
}

} // namespace rund::compute::detail
