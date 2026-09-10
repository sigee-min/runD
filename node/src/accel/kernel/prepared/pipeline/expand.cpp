#include "expand.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

namespace rund::node::accel::detail {

namespace {

[[nodiscard]] bool append_pipeline_occurrence(
    const std::span<const BackendBatchEntry> templates,
    ExpandedPipeline &expanded, const std::size_t template_index,
    const std::uint8_t barrier, const std::uint32_t outer,
    const std::uint32_t outer_bound, const std::uint32_t inner,
    const std::uint32_t inner_bound, const std::uint32_t route,
    const std::uint32_t transducer = NoTileTransducer,
    const std::uint32_t inner_advance = NoTileTransducer) {
  if (template_index >= templates.size() ||
      expanded.commands.size() >= std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  expanded.failure.compact_template_route(
      static_cast<std::uint32_t>(template_index),
      templates[template_index].recurrence);
  BackendBatchEntry command = templates[template_index];
  command.template_index = static_cast<std::uint32_t>(template_index);
  command.occurrence_index =
      static_cast<std::uint32_t>(expanded.commands.size());
  command.transducer = transducer;
  if (command.recurrence.window != nullptr) {
    expanded.windows.push_back(*command.recurrence.window);
    BackendWindow &window = expanded.windows.back();
    window.outer_iteration = outer;
    window.outer_bound = outer_bound;
    window.inner_iteration = inner;
    window.inner_bound = inner_bound;
    window.inner_advance =
        inner_advance != NoTileTransducer
            ? inner_advance
            : (window.phase == BackendWindowPhase::NestedAction ? 1u : 0u);
    window.route = route;
    command.recurrence.window = &window;
  }
  expanded.commands.push_back(std::move(command));
  expanded.barriers.push_back(barrier);
  expanded.failure.occurrence_route(expanded.commands.back());
  return true;
}

} // namespace

[[nodiscard]] bool expand_pipeline(
    const std::span<const BackendBatchEntry> templates,
    const std::span<const std::uint8_t> template_barriers,
    const std::span<const BackendPublish> publications,
    const std::span<const std::uint32_t> declared_steps,
    const std::uint32_t declared_step_count, const bool profile_steps,
    const std::uint32_t direct_aggregate_commands, ExpandedPipeline &expanded) {
  expanded.failure.stage(PreparedPipelineFailureStage::CommonExpansion);
  if (templates.empty() || templates.size() != template_barriers.size() ||
      templates.size() != declared_steps.size()) {
    return false;
  }
  std::vector<std::uint32_t> group_transducers;
  std::uint64_t command_count = 0u;
  for (std::size_t index = 0u; index < templates.size();) {
    expanded.failure.template_route(static_cast<std::uint32_t>(index));
    const BackendWindow *const window = templates[index].recurrence.window;
    if (window == nullptr || window->phase == BackendWindowPhase::Ordinary) {
      ++command_count;
      ++index;
      continue;
    }
    NestedTemplateGeometry geometry{};
    if (!ProveNestedTemplateGeometry(templates, index, geometry)) {
      return false;
    }
    const std::size_t end = geometry.end();
    const NestedTemplateShape &shape = geometry.shape();
    const std::uint32_t inner_bound = geometry.inner_bound();
    NestedAggregate aggregate =
        BuildNestedAggregate(templates, template_barriers, publications, index);
    if (aggregate.invalid()) {
      expanded.reason = aggregate.reason;
      return false;
    }
    if (aggregate.ready()) {
      try {
        expanded.aggregates.push_back(std::move(aggregate));
      } catch (const std::bad_alloc &) {
        expanded.reason = "compute_pipeline_capacity";
        return false;
      }
    }
    const NestedAggregate *const direct = expanded.aggregates.size() == 1u
                                              ? &expanded.aggregates.front()
                                              : nullptr;
    bool declared_seed_range =
        direct != nullptr && direct->shape.seed_first() == 0u;
    if (declared_seed_range) {
      const std::size_t seed_first = direct->shape.seed_first();
      const std::uint32_t first = declared_steps[seed_first];
      for (std::uint32_t outer = 0u; outer < direct->shape.seed_count();
           ++outer) {
        if (first > std::numeric_limits<std::uint32_t>::max() - outer ||
            declared_steps[seed_first + outer] != first + outer) {
          declared_seed_range = false;
          break;
        }
      }
    }
    bool profile_layout = !profile_steps;
    if (profile_steps && declared_step_count == templates.size()) {
      profile_layout = true;
      for (std::size_t declared = 0u; declared < declared_steps.size();
           ++declared) {
        if (declared_steps[declared] != declared) {
          profile_layout = false;
          break;
        }
      }
    }
    const bool complete_direct =
        direct_aggregate_commands != 0u && direct != nullptr && index == 0u &&
        end == templates.size() && direct->shape == shape &&
        direct->shape.first() == 0u &&
        direct->shape.end() == templates.size() && publications.size() == 1u &&
        direct->publication_index == 0u &&
        direct->failure.logical_step == declared_steps.front() &&
        direct->profile.aggregate_profile_supported && declared_seed_range &&
        profile_layout;
    if (complete_direct) {
      // The native aggregate consumes compact templates directly. Retaining
      // K*(N+2) occurrence descriptors, barriers, and copied window records
      // would recreate the intermediate memory layer this proof eliminates.
      expanded.command_count = direct_aggregate_commands;
      expanded.compact_aggregate = true;
      return true;
    }
    if (group_transducers.empty()) {
      group_transducers.assign(templates.size(), NoTileTransducer);
    }
    const std::size_t action_first = geometry.action_first();
    MapRecurrence recurrence = BuildNestedMapRecurrence(
        templates.subspan(action_first, inner_bound),
        template_barriers.subspan(action_first, inner_bound), geometry);
    if (recurrence.invalid()) {
      expanded.reason = recurrence.reason;
      return false;
    }
    const bool fused = recurrence.ready();
    if (fused) {
      // Proof retains only the canonical Program pointer plus its exact source
      // recipe. No transformed artifact survives expansion, so nested group
      // count cannot recreate an intermediate source/metadata memory layer.
      if (recurrence.canonical_artifact == nullptr ||
          !recurrence.source_plan.ok || recurrence.history != nullptr) {
        expanded.reason = "accel_kernel_run_invalid";
        return false;
      }
      if (expanded.transducers.size() >= NoTileTransducer) {
        expanded.reason = "compute_pipeline_capacity";
        return false;
      }
      group_transducers[index] =
          static_cast<std::uint32_t>(expanded.transducers.size());
      expanded.transducers.push_back(TileTransducer{
          .recurrence = std::move(recurrence),
          .template_first = static_cast<std::uint32_t>(action_first),
          .template_count = inner_bound,
      });
    }
    const std::uint64_t commands = fused ? shape.transduced_occurrence_count()
                                         : shape.authored_occurrence_count();
    if (command_count > std::numeric_limits<std::uint32_t>::max() ||
        commands > std::numeric_limits<std::uint32_t>::max() - command_count) {
      return false;
    }
    command_count += commands;
    index = end;
  }
  if (command_count == 0u ||
      command_count > std::numeric_limits<std::uint32_t>::max() ||
      command_count > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  expanded.command_count = static_cast<std::uint32_t>(command_count);

  const std::size_t capacity = static_cast<std::size_t>(command_count);
  expanded.commands.reserve(capacity);
  expanded.barriers.reserve(capacity);
  expanded.windows.reserve(capacity);

  for (std::size_t index = 0u; index < templates.size();) {
    expanded.failure.template_route(static_cast<std::uint32_t>(index));
    const BackendWindow *const window = templates[index].recurrence.window;
    if (window == nullptr || window->phase == BackendWindowPhase::Ordinary) {
      const std::uint32_t iteration =
          window == nullptr ? 0u : window->outer_iteration;
      const std::uint32_t bound = window == nullptr ? 1u : window->outer_bound;
      if (!append_pipeline_occurrence(templates, expanded, index,
                                      template_barriers[index], iteration,
                                      bound, 0u, 1u, 0u)) {
        return false;
      }
      ++index;
      continue;
    }

    NestedTemplateGeometry geometry{};
    if (!ProveNestedTemplateGeometry(templates, index, geometry)) {
      return false;
    }
    const std::size_t end = geometry.end();
    const NestedTemplateShape &shape = geometry.shape();
    const std::uint32_t outer_bound = geometry.outer_bound();
    const std::uint32_t inner_bound = geometry.inner_bound();
    const std::size_t action_first = geometry.action_first();
    const std::size_t fold_first = geometry.fold_first();
    const std::uint32_t transducer = group_transducers[index];
    for (std::uint32_t outer = 0u; outer < outer_bound; ++outer) {
      const bool first_command = expanded.commands.empty();
      const std::uint8_t seed_barrier =
          first_command ? template_barriers[index + outer] : 1u;
      if (!append_pipeline_occurrence(templates, expanded, index + outer,
                                      seed_barrier, outer, outer_bound, 0u,
                                      inner_bound, 0u)) {
        return false;
      }
      if (transducer != NoTileTransducer) {
        if (!append_pipeline_occurrence(templates, expanded, action_first, 1u,
                                        outer, outer_bound, 0u, inner_bound, 0u,
                                        transducer, 0u)) {
          return false;
        }
      } else {
        for (std::uint32_t inner = 0u; inner < inner_bound; ++inner) {
          if (!append_pipeline_occurrence(
                  templates, expanded, action_first + inner, 1u, outer,
                  outer_bound, inner, inner_bound, 0u)) {
            return false;
          }
        }
      }
      std::uint32_t route = 0u;
      if (!shape.fold_route_for_outer(outer, route)) {
        return false;
      }
      if (!append_pipeline_occurrence(
              templates, expanded, fold_first + route, 1u, outer, outer_bound,
              inner_bound, inner_bound, route, NoTileTransducer,
              transducer == NoTileTransducer ? 0u : inner_bound)) {
        return false;
      }
    }
    index = end;
  }
  return expanded.commands.size() == capacity &&
         expanded.barriers.size() == capacity;
}

bool MaterializePreparedPipelineOccurrenceForContract(
    const BackendWindow &template_window, const std::uint32_t outer,
    const std::uint32_t outer_bound, const std::uint32_t inner,
    const std::uint32_t inner_bound, const std::uint32_t route,
    const std::uint32_t inner_advance, const bool transduced,
    PreparedPipelineFailure &failure) noexcept {
  try {
    const std::array<BackendBatchEntry, 1u> templates{{
        {.recurrence = BackendRecurrence{.window = &template_window}},
    }};
    ExpandedPipeline expanded{};
    expanded.commands.reserve(1u);
    expanded.barriers.reserve(1u);
    expanded.windows.reserve(1u);
    expanded.failure.stage(PreparedPipelineFailureStage::CommonExpansion);
    if (!append_pipeline_occurrence(
            templates, expanded, 0u, 0u, outer, outer_bound, inner, inner_bound,
            route, transduced ? 0u : NoTileTransducer, inner_advance) ||
        expanded.commands.size() != 1u || expanded.windows.size() != 1u ||
        expanded.commands.front().recurrence.window !=
            &expanded.windows.front()) {
      return false;
    }
    failure = expanded.failure.failure("contract_occurrence");
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

} // namespace rund::node::accel::detail
