#include "../../state/assembly.hpp"
#include "model.hpp"

#include "../../claim.hpp"
#include "../../local.hpp"

#include <cstdint>
#include <limits>
#include <utility>

namespace rund::compute::detail {

using admit::admit_resolved_view;

Status admit_state_pairs(AdmissionDraft &draft) {
  PipelineState *const state = draft.state.get();
  const PipelineMemoryPlan &plan = draft.plan;
  const PipelineBuildSnapshot &frozen = draft.frozen;
  const PipelineBuildState *const build = &draft.build;
  auto &hash = draft.hash;
  const std::size_t resource_count = draft.resource_count;

  state->transactional = frozen.commit;
  state->publication->state_pairs.reserve(plan.state_pair_resources.size());
  hash.number(plan.state_pair_resources.size());
  for (const PipelineStatePairResourcePlan &pair : plan.state_pair_resources) {
    if (pair.published.resource >= plan.resources.size() ||
        pair.pending.resource >= plan.resources.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const PipelineResolvedResourcePlan &published_plan =
        plan.resources[pair.published.resource];
    const PipelineResolvedResourcePlan &pending_plan =
        plan.resources[pair.pending.resource];
    auto published =
        admit_resolved_view(plan, *state, pair.published, published_plan.type,
                            static_cast<std::size_t>(published_plan.count),
                            published_plan.format, ResourceAccess::Read);
    auto pending =
        admit_resolved_view(plan, *state, pair.pending, pending_plan.type,
                            static_cast<std::size_t>(pending_plan.count),
                            pending_plan.format, ResourceAccess::Write);
    if (!published || !pending || *published == *pending) {
      return Status::fail(!published ? published.reason()
                          : !pending ? pending.reason()
                                     : Reason::PipelineInvalid);
    }
    PipelineResource &published_resource = state->resources[*published];
    PipelineResource &pending_resource = state->resources[*pending];
    if (*published != pair.published.resource ||
        *pending != pair.pending.resource) {
      return Status::fail(Reason::PipelineInvalid);
    }
    // A state pair shares one typed storage schema, not one Program arithmetic
    // policy. Fixed rounding/overflow/approximation remain canonical on each
    // resource's producing/consuming Program; parity only swaps equal-width
    // storage owners, as the public state contract has always allowed.
    if (pair.published.offset != 0u || pair.published.stride != 1u ||
        pair.published.count != published_plan.count ||
        pair.pending.offset != 0u || pair.pending.stride != 1u ||
        pair.pending.count != pending_plan.count ||
        published_resource.type != pending_resource.type ||
        published_resource.count != pending_resource.count ||
        published_resource.bytes != pending_resource.bytes ||
        published_plan.output ||
        pair.pending_first_full_write == resource::NoNode ||
        pair.pending_first_input <= pair.pending_first_full_write ||
        published_resource.partner != PipelineResource::no_output ||
        pending_resource.partner != PipelineResource::no_output) {
      return Status::fail(Reason::PipelineInvalid);
    }
    published_resource.partner = *pending;
    pending_resource.partner = *published;
    state->publication->state_pairs.push_back(PipelineStatePair{
        .first = build->materialized_resources[*published],
        .second = build->materialized_resources[*pending],
        .type = published_resource.type,
        .format = published_resource.format,
        .count = published_resource.count,
        .bytes = published_resource.bytes,
    });
    hash.number(*published);
    hash.number(*pending);
    hash.number(static_cast<std::uint64_t>(published_resource.type));
    hash.format(published_resource.format);
    hash.number(published_resource.count);
    hash.number(published_resource.bytes);
  }

  if (state->resources.size() != resource_count) {
    return Status::fail(Reason::PipelineInvalid);
  }

  return Status::success();
}

} // namespace rund::compute::detail
