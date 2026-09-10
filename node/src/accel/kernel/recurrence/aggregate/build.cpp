#include "internal.hpp"

#include <limits>
#include <utility>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] NestedAggregate Ineligible(const char *const reason) noexcept {
  return NestedAggregate{.reason = reason};
}

[[nodiscard]] NestedAggregate Invalid(const char *const reason) noexcept {
  return NestedAggregate{
      .state = NestedAggregateState::Invalid,
      .reason = reason,
  };
}

} // namespace

NestedAggregate
BuildNestedAggregate(const std::span<const BackendBatchEntry> templates,
                     const std::span<const std::uint8_t> barriers,
                     const std::span<const BackendPublish> publications,
                     const std::size_t first) {
  using namespace nested_aggregate_detail;

  if (barriers.size() != templates.size()) {
    return Invalid("compute_pipeline_nested_aggregate_barrier_invalid");
  }
  NestedTemplateGeometry geometry{};
  if (!ExactAggregateEnvelope(templates, barriers, first, geometry)) {
    return Ineligible("compute_pipeline_nested_aggregate_shape_ineligible");
  }
  const NestedTemplateShape &shape = geometry.shape();
  const BackendWindow &window = *geometry.window();

  const BackendRun *const seed_run = templates[shape.seed_first()].run;
  if (seed_run == nullptr) {
    return Ineligible("compute_pipeline_nested_aggregate_seed_ineligible");
  }
  const ProgramFingerprint seed_program = ProgramIdentity(*seed_run);
  if (!U32Program(seed_program)) {
    return Ineligible("compute_pipeline_nested_aggregate_seed_ineligible");
  }
  SeedProjection seed_projection{};
  const char *seed_reason = "compute_pipeline_nested_aggregate_seed_ineligible";
  for (std::uint32_t outer = 0u; outer < shape.seed_count(); ++outer) {
    const BackendRun *const run = templates[shape.seed_first() + outer].run;
    SeedProjection current{};
    if (run == nullptr || !SameProgram(*run, seed_program)) {
      return Ineligible(
          "compute_pipeline_nested_aggregate_seed_identity_ineligible");
    }
    if (!SeedShape(*run, window, current, seed_reason)) {
      return Ineligible(seed_reason);
    }
    if (outer != 0u &&
        (!SameRead(seed_projection.queue, current.queue) ||
         !SameRead(seed_projection.domain, current.domain) ||
         !SameRead(seed_projection.count, current.count) ||
         !SameStorage(seed_projection.tile_low, current.tile_low) ||
         !SameStorage(seed_projection.tile_status, current.tile_status) ||
         !SameStorage(seed_projection.tile_state, current.tile_state) ||
         !SameStorage(seed_projection.tile_count, current.tile_count) ||
         seed_projection.invalid_index_source_node !=
             current.invalid_index_source_node ||
         seed_projection.reduce_overflow_source_node !=
             current.reduce_overflow_source_node)) {
      return Ineligible(
          "compute_pipeline_nested_aggregate_seed_identity_ineligible");
    }
    if (outer == 0u) {
      seed_projection = std::move(current);
    }
  }

  ProgramFingerprint action_program{};
  NestedScalarExpr action_expression{};
  View action_state{};
  std::uint32_t action_dispatches{};
  if (!BuildAction(templates, shape, seed_projection, action_program,
                   action_expression, action_state, action_dispatches)) {
    return Ineligible("compute_pipeline_nested_aggregate_action_ineligible");
  }

  BackendPublish publication{};
  std::uint32_t publication_index = NoNode;
  if (!PublicationFor(publications, window, publication, publication_index)) {
    return Ineligible(
        "compute_pipeline_nested_aggregate_publication_ineligible");
  }
  ProgramFingerprint fold_program{};
  NestedScalarExpr fold_expression{};
  std::uint32_t fold_dispatches{};
  if (!BuildFold(templates, shape, seed_projection, action_state, publication,
                 fold_program, fold_expression, fold_dispatches)) {
    return Ineligible("compute_pipeline_nested_aggregate_fold_ineligible");
  }

  const std::uint64_t seed_dispatches = seed_run->final_dispatch_count;
  if (seed_dispatches == 0u ||
      seed_dispatches > std::numeric_limits<std::uint32_t>::max()) {
    return Ineligible("compute_pipeline_nested_aggregate_profile_ineligible");
  }
  return NestedAggregate{
      .state = NestedAggregateState::Ready,
      .shape = shape,
      .maximum = window.maximum,
      .tile = window.tile,
      .queue =
          NestedAggregateRead{
              .read = std::move(seed_projection.queue),
              .logical_count = seed_projection.queue.source.count,
              .element_bytes = seed_projection.queue.source.element_bytes,
          },
      .domain =
          NestedAggregateRead{
              .read = std::move(seed_projection.domain),
              .logical_count = seed_projection.domain.source.count,
              .element_bytes = seed_projection.domain.source.element_bytes,
          },
      .count =
          NestedAggregateRead{
              .read = std::move(seed_projection.count),
              .logical_count = 1u,
              .element_bytes = sizeof(std::uint32_t),
          },
      .tile_low = Workspace(seed_projection.tile_low),
      .tile_status = Workspace(seed_projection.tile_status),
      .action_expr = action_expression,
      .fold_expr = fold_expression,
      .publication = std::move(publication),
      .publication_index = publication_index,
      .failure =
          NestedAggregateFailureProjection{
              .logical_step =
                  templates[shape.seed_first()].recurrence.logical_step,
              .invalid_index_source_node =
                  seed_projection.invalid_index_source_node,
              .reduce_overflow_source_node =
                  seed_projection.reduce_overflow_source_node,
              .count_overflow_reason = static_cast<std::uint32_t>(
                  rund::compute::Reason::BoundedCountInvalid),
              .invalid_index_reason = static_cast<std::uint32_t>(
                  rund::compute::Reason::GatherIndexOutOfRange),
              .reduce_overflow_reason = static_cast<std::uint32_t>(
                  rund::compute::Reason::ReduceSumOverflow),
              .phase = rund::compute::PipelineNestedPhase::Seed,
              .inner_coordinate_unknown = true,
          },
      .profile =
          NestedAggregateProfileProjection{
              .seed_dispatches_per_occurrence =
                  static_cast<std::uint32_t>(seed_dispatches),
              .action_dispatches_per_occurrence = action_dispatches,
              .fold_dispatches_per_occurrence = fold_dispatches,
              .aggregate_profile_supported = true,
          },
      .reason = "ok",
  };
}

} // namespace rund::node::accel::detail
