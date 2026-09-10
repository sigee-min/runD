#include "../state/assembly.hpp"
#include "prepare.hpp"
#include "admit/model.hpp"

#include <memory>

namespace rund::compute::detail {

Status admit_pipeline(const std::shared_ptr<PipelineBuildState> &build,
                      PipelinePrepare &prepare) {
  if (build->memory == nullptr || build->memory->frozen == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const PipelineMemoryPlan &plan = *build->memory;
  const PipelineBuildSnapshot &frozen = *plan.frozen;
  AdmissionDraft draft{*build, plan, frozen};
  // Preserve the canonical hash prefix owned by the admission coordinator.
  // The split leaves append step/publication/state-pair fields to their phase
  // owners, but sealed repetition and step count precede those fields.
  draft.hash.number(frozen.sealed_repetitions);
  draft.hash.number(frozen.steps.size());

  const Status initial = admit_initial(draft);
  if (!initial) {
    return initial;
  }
  const Status steps = admit_steps(draft);
  if (!steps) {
    return steps;
  }
  const Status publications = admit_publications(draft);
  if (!publications) {
    return publications;
  }
  const Status pairs = admit_state_pairs(draft);
  if (!pairs) {
    return pairs;
  }

  prepare.state = std::move(draft.state);
  prepare.hash = draft.hash;
  prepare.output_count = draft.output_count;
  prepare.status_count = draft.status_entry_count;
  return Status::success();
}

} // namespace rund::compute::detail
