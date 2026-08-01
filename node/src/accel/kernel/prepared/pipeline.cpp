#include "../prepared.hpp"

#include "../recurrence.hpp"
#include "evidence.hpp"
#include "model.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <vector>

namespace rund::node::accel::detail {
namespace {

struct ExpandedPipeline final {
  std::vector<BackendBatchEntry> commands;
  std::vector<std::uint8_t> barriers;
  std::vector<BackendWindow> windows;
  std::vector<TileTransducer> transducers;
  std::vector<NestedAggregate> aggregates;
  std::uint32_t command_count{};
  bool compact_aggregate{};
  const char *reason = "accel_kernel_run_invalid";
};

[[nodiscard]] bool same_nested_window(const BackendWindow &left,
                                      const BackendWindow &right) noexcept {
  return left.state == right.state && left.maximum == right.maximum &&
         left.tile == right.tile && left.expected == right.expected &&
         left.outer_bound == right.outer_bound &&
         left.inner_bound == right.inner_bound &&
         left.has_terminal == right.has_terminal &&
         left.count.source.id == right.count.source.id &&
         left.count.source.offset_bytes == right.count.source.offset_bytes &&
         left.count.handle == right.count.handle;
}

[[nodiscard]] bool
nested_shape(const std::span<const BackendBatchEntry> templates,
             const std::size_t first, std::size_t &end,
             std::uint32_t &outer_bound, std::uint32_t &inner_bound) noexcept {
  if (first >= templates.size()) {
    return false;
  }
  const BackendWindow *const first_window = templates[first].recurrence.window;
  if (first_window == nullptr ||
      first_window->phase != BackendWindowPhase::NestedSeed ||
      first_window->outer_bound == 0u) {
    return false;
  }
  outer_bound = first_window->outer_bound;
  inner_bound = first_window->inner_bound;
  const std::uint64_t group_count =
      static_cast<std::uint64_t>(outer_bound) + inner_bound + 3u;
  if (group_count > templates.size() - first) {
    return false;
  }
  end = first + static_cast<std::size_t>(group_count);
  for (std::uint32_t outer = 0u; outer < outer_bound; ++outer) {
    const BackendWindow *const window =
        templates[first + outer].recurrence.window;
    if (window == nullptr || window->phase != BackendWindowPhase::NestedSeed ||
        window->outer_iteration != outer || window->route != 0u ||
        !same_nested_window(*first_window, *window)) {
      return false;
    }
  }
  const std::size_t action_first = first + outer_bound;
  for (std::uint32_t inner = 0u; inner < inner_bound; ++inner) {
    const BackendWindow *const window =
        templates[action_first + inner].recurrence.window;
    if (window == nullptr ||
        window->phase != BackendWindowPhase::NestedAction ||
        window->inner_iteration != inner || window->route != 0u ||
        !same_nested_window(*first_window, *window)) {
      return false;
    }
  }
  const std::size_t fold_first = action_first + inner_bound;
  for (std::uint32_t route = 0u; route < 3u; ++route) {
    const BackendWindow *const window =
        templates[fold_first + route].recurrence.window;
    if (window == nullptr || window->phase != BackendWindowPhase::NestedFold ||
        window->route != route || !same_nested_window(*first_window, *window)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool expand_pipeline(
    const std::span<const BackendBatchEntry> templates,
    const std::span<const std::uint8_t> template_barriers,
    const std::span<const BackendPublish> publications,
    const std::span<const std::uint32_t> declared_steps,
    const std::uint32_t declared_step_count, const bool profile_steps,
    const std::uint32_t direct_aggregate_commands, ExpandedPipeline &expanded) {
  if (templates.empty() || templates.size() != template_barriers.size() ||
      templates.size() != declared_steps.size()) {
    return false;
  }
  std::vector<std::uint32_t> group_transducers;
  std::uint64_t command_count = 0u;
  for (std::size_t index = 0u; index < templates.size();) {
    const BackendWindow *const window = templates[index].recurrence.window;
    if (window == nullptr || window->phase == BackendWindowPhase::Ordinary) {
      ++command_count;
      ++index;
      continue;
    }
    std::size_t end = 0u;
    std::uint32_t outer_bound = 0u;
    std::uint32_t inner_bound = 0u;
    if (window->phase != BackendWindowPhase::NestedSeed ||
        !nested_shape(templates, index, end, outer_bound, inner_bound)) {
      return false;
    }
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
    bool declared_seed_range = direct != nullptr && direct->seed.first == 0u;
    if (declared_seed_range) {
      const std::uint32_t first = declared_steps[direct->seed.first];
      for (std::uint32_t outer = 0u; outer < direct->seed.count; ++outer) {
        if (first > std::numeric_limits<std::uint32_t>::max() - outer ||
            declared_steps[direct->seed.first + outer] != first + outer) {
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
        end == templates.size() && direct->seed.count == outer_bound &&
        direct->action.first == direct->seed.end() &&
        direct->action.count == inner_bound &&
        direct->fold.first == direct->action.end() &&
        direct->fold.count == 3u && direct->fold.end() == templates.size() &&
        publications.size() == 1u && direct->publication_index == 0u &&
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
    const std::size_t action_first = index + outer_bound;
    MapRecurrence recurrence = BuildNestedMapRecurrence(
        templates.subspan(action_first, inner_bound),
        template_barriers.subspan(action_first, inner_bound));
    if (recurrence.invalid()) {
      expanded.reason = recurrence.reason;
      return false;
    }
    const bool fused = recurrence.ready();
    if (fused) {
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
    const std::uint64_t commands_per_outer =
        fused ? 3u : static_cast<std::uint64_t>(inner_bound) + 2u;
    if (command_count > std::numeric_limits<std::uint32_t>::max() ||
        (commands_per_outer != 0u &&
         outer_bound >
             (std::numeric_limits<std::uint32_t>::max() - command_count) /
                 commands_per_outer)) {
      return false;
    }
    command_count +=
        static_cast<std::uint64_t>(outer_bound) * commands_per_outer;
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
  const auto append = [&](const std::size_t template_index,
                          const std::uint8_t barrier, const std::uint32_t outer,
                          const std::uint32_t outer_bound,
                          const std::uint32_t inner,
                          const std::uint32_t inner_bound,
                          const std::uint32_t route,
                          const std::uint32_t transducer = NoTileTransducer,
                          const std::uint32_t inner_advance =
                              NoTileTransducer) {
    if (template_index >= templates.size() ||
        expanded.commands.size() >= std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    BackendBatchEntry command = templates[template_index];
    command.template_index = static_cast<std::uint32_t>(template_index);
    command.occurrence_index =
        static_cast<std::uint32_t>(expanded.commands.size());
    command.transducer = transducer;
    if (command.recurrence.window != nullptr) {
      expanded.windows.push_back(*command.recurrence.window);
      BackendWindow &window = expanded.windows.back();
      window.iteration = outer;
      window.bound = outer_bound;
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
    return true;
  };

  for (std::size_t index = 0u; index < templates.size();) {
    const BackendWindow *const window = templates[index].recurrence.window;
    if (window == nullptr || window->phase == BackendWindowPhase::Ordinary) {
      const std::uint32_t iteration =
          window == nullptr ? 0u : window->outer_iteration;
      const std::uint32_t bound = window == nullptr ? 1u : window->outer_bound;
      if (!append(index, template_barriers[index], iteration, bound, 0u, 1u,
                  0u)) {
        return false;
      }
      ++index;
      continue;
    }

    std::size_t end = 0u;
    std::uint32_t outer_bound = 0u;
    std::uint32_t inner_bound = 0u;
    if (!nested_shape(templates, index, end, outer_bound, inner_bound)) {
      return false;
    }
    const std::size_t action_first = index + outer_bound;
    const std::size_t fold_first = action_first + inner_bound;
    const std::uint32_t transducer = group_transducers[index];
    for (std::uint32_t outer = 0u; outer < outer_bound; ++outer) {
      const bool first_command = expanded.commands.empty();
      const std::uint8_t seed_barrier =
          first_command ? template_barriers[index + outer] : 1u;
      if (!append(index + outer, seed_barrier, outer, outer_bound, 0u,
                  inner_bound, 0u)) {
        return false;
      }
      if (transducer != NoTileTransducer) {
        if (!append(action_first, 1u, outer, outer_bound, 0u, inner_bound, 0u,
                    transducer, 0u)) {
          return false;
        }
      } else {
        for (std::uint32_t inner = 0u; inner < inner_bound; ++inner) {
          if (!append(action_first + inner, 1u, outer, outer_bound, inner,
                      inner_bound, 0u)) {
            return false;
          }
        }
      }
      const std::uint32_t route =
          outer == 0u ? 0u : ((outer & 1u) != 0u ? 1u : 2u);
      if (!append(fold_first + route, 1u, outer, outer_bound, inner_bound,
                  inner_bound, route, NoTileTransducer,
                  transducer == NoTileTransducer ? 0u : inner_bound)) {
        return false;
      }
    }
    index = end;
  }
  return expanded.commands.size() == capacity &&
         expanded.barriers.size() == capacity;
}

} // namespace

PreparedKernelPipeline
PrepareKernelPipeline(const rund::AccelContext &context,
                      const std::span<const PreparedKernelRun *const> runs,
                      const std::span<const std::uint8_t> barriers,
                      const std::span<const std::uint32_t> declared_steps,
                      const std::span<const BackendRecurrence> recurrences,
                      const std::span<const BackendPublish> publications,
                      const std::uint32_t declared_step_count,
                      const std::uint32_t generation_stride,
                      const bool profile_steps) {
  const rund::AccelCheck invalid{false, "accel_kernel_run_invalid"};
  if (runs.empty() || runs.size() > PreparedPipelineStepCapacity ||
      runs.size() != barriers.size() || runs.size() != declared_steps.size() ||
      runs.size() != recurrences.size() || publications.size() > 32u) {
    return {};
  }
  std::uint32_t state_count = 0u;
  for (const BackendRecurrence &recurrence : recurrences) {
    if (recurrence.window != nullptr) {
      state_count = std::max(state_count, recurrence.window->state + 1u);
    }
  }
  for (const BackendPublish &publication : publications) {
    const auto &target = publication.target;
    const bool window = publication.kind == BackendPublishKind::Window;
    if (publication.target_handle == nullptr ||
        publication.state >= state_count ||
        (!window && publication.final >= publication.sources.size()) ||
        (window &&
         (publication.maximum == 0u || publication.tile == 0u ||
          publication.tile > publication.maximum ||
          target.count != publication.maximum ||
          publication.count.handle == nullptr ||
          publication.count.source.count != 1u ||
          publication.count.source.element_bytes != sizeof(std::uint32_t) ||
          publication.count.source.stride_bytes < sizeof(std::uint32_t) ||
          publication.count.source.usage !=
              rund::kernel::kResidentUsageRead)) ||
        (!window && (publication.maximum != 0u || publication.tile != 0u)) ||
        target.stride_bytes < target.element_bytes ||
        target.usage != rund::kernel::kResidentUsageWrite) {
      return PreparedKernelPipeline{.reason = invalid.reason};
    }
    for (const BackendRead &read : publication.sources) {
      const auto &source = read.source;
      if (read.handle == nullptr ||
          source.count != (window ? publication.tile : target.count) ||
          source.element_bytes != target.element_bytes ||
          (source.element_bytes != 4u && source.element_bytes != 8u) ||
          source.stride_bytes < source.element_bytes ||
          source.usage != rund::kernel::kResidentUsageRead) {
        return PreparedKernelPipeline{.reason = invalid.reason};
      }
    }
  }
  std::shared_ptr<prepared::PipelineState> pipeline{};
  try {
    pipeline = std::make_shared<prepared::PipelineState>();
    pipeline->states =
        std::make_unique<std::shared_ptr<prepared::RunState>[]>(runs.size());
  } catch (const std::bad_alloc &) {
    return {};
  }
  pipeline->context = context;
  pipeline->size = runs.size();
  std::vector<BackendBatchEntry> templates;
  ExpandedPipeline expanded;
  try {
    templates.resize(runs.size());
    for (std::size_t index = 0u; index < runs.size(); ++index) {
      const PreparedKernelRun *const item = runs[index];
      auto *const state =
          item == nullptr
              ? nullptr
              : static_cast<prepared::RunState *>(item->owner.get());
      const BackendOps *const candidate =
          state == nullptr ? nullptr : state->bound.run.ops;
      if (item == nullptr || !item->ok || state == nullptr ||
          !IsPipelinePrivatePreparation(state->mode) || candidate == nullptr ||
          candidate->prepare_pipeline == nullptr ||
          candidate->submit_prepared_pipeline == nullptr ||
          !prepared::MatchesContext(context, *state) ||
          (pipeline->ops != nullptr && pipeline->ops != candidate)) {
        return PreparedKernelPipeline{.reason = invalid.reason};
      }
      pipeline->ops = candidate;
      pipeline->states[index] =
          std::static_pointer_cast<prepared::RunState>(item->owner);
      templates[index] = BackendBatchEntry{
          .run = &state->bound.run,
          .prepared = &state->backend,
          .recurrence = recurrences[index],
          .template_index = static_cast<std::uint32_t>(index)};
    }
    if (!expand_pipeline(templates, barriers, publications, declared_steps,
                         declared_step_count, profile_steps,
                         pipeline->ops->nested_aggregate_command_count,
                         expanded)) {
      return PreparedKernelPipeline{.reason = expanded.reason};
    }
  } catch (const std::bad_alloc &) {
    return PreparedKernelPipeline{.reason = "compute_pipeline_capacity"};
  }
  const auto account_current = [&]() -> const char * {
    pipeline->counts = {};
    if (expanded.command_count == 0u ||
        !PreparePipelineStatusLayout(
            pipeline->status, declared_steps, declared_step_count,
            expanded.command_count, generation_stride)) {
      return invalid.reason;
    }
    if (expanded.compact_aggregate) {
      if (expanded.aggregates.size() != 1u) {
        return invalid.reason;
      }
      const NestedAggregate &aggregate = expanded.aggregates.front();
      for (std::size_t index = 0u; index < pipeline->size; ++index) {
        const std::uint64_t occurrences = aggregate.authored_occurrences(index);
        if (pipeline->states[index] == nullptr || occurrences == 0u) {
          return invalid.reason;
        }
        prepared::Accumulate(pipeline->counts, *pipeline->states[index],
                             occurrences);
      }
      return nullptr;
    }
    for (const BackendBatchEntry &command : expanded.commands) {
      if (command.template_index >= pipeline->size ||
          pipeline->states[command.template_index] == nullptr) {
        return invalid.reason;
      }
      if (command.transducer == NoTileTransducer) {
        prepared::Accumulate(pipeline->counts,
                             *pipeline->states[command.template_index]);
        continue;
      }
      if (command.transducer >= expanded.transducers.size()) {
        return invalid.reason;
      }
      const TileTransducer &transducer =
          expanded.transducers[command.transducer];
      const std::uint64_t template_end =
          static_cast<std::uint64_t>(transducer.template_first) +
          transducer.template_count;
      if (transducer.template_count == 0u || template_end > pipeline->size) {
        return invalid.reason;
      }
      // One physical transducer command represents the complete authored
      // Action subrange for this outer window. Evidence remains logical: count
      // every original template once while the backend reports the smaller
      // physical dispatch count independently.
      for (std::uint32_t offset = 0u; offset < transducer.template_count;
           ++offset) {
        const std::size_t template_index = transducer.template_first + offset;
        if (pipeline->states[template_index] == nullptr) {
          return invalid.reason;
        }
        prepared::Accumulate(pipeline->counts,
                             *pipeline->states[template_index]);
      }
    }
    return nullptr;
  };

  bool canonical_fallback = false;
  for (;;) {
    if (const char *const reason = account_current(); reason != nullptr) {
      return PreparedKernelPipeline{.reason = reason};
    }
    const rund::AccelCheck built = pipeline->ops->prepare_pipeline(
        templates, expanded.commands, expanded.barriers, expanded.transducers,
        expanded.aggregates, publications, pipeline->status, profile_steps,
        pipeline->backend, pipeline->memory);
    if (built.ok) {
      break;
    }
    if (!expanded.compact_aggregate || canonical_fallback) {
      return PreparedKernelPipeline{.reason = built.reason};
    }

    // Native aggregation is an optional materialization of the common proof.
    // If device admission or compilation rejects it, lazily build the
    // canonical occurrence stream once. The successful hot path never owns
    // that intermediate representation, while valid Pipelines retain their
    // backend-neutral fallback semantics.
    ExpandedPipeline canonical;
    try {
      if (!expand_pipeline(templates, barriers, publications, declared_steps,
                           declared_step_count, profile_steps, 0u, canonical)) {
        return PreparedKernelPipeline{.reason = canonical.reason};
      }
    } catch (const std::bad_alloc &) {
      return PreparedKernelPipeline{.reason = "compute_pipeline_capacity"};
    }
    canonical.aggregates.clear();
    expanded = std::move(canonical);
    pipeline->backend.reset();
    pipeline->memory = {};
    canonical_fallback = true;
  }
  if (!ValidPreparedPipelineStatusLayout(
          pipeline->status, declared_steps, declared_step_count,
          expanded.command_count, generation_stride)) {
    return PreparedKernelPipeline{.reason = invalid.reason};
  }
  const std::uint64_t common_host_bytes =
      sizeof(prepared::PipelineState) +
      pipeline->size * sizeof(std::shared_ptr<prepared::RunState>);
  accumulate_memory(pipeline->memory.host,
                    PreparedMemory{.current = common_host_bytes,
                                   .peak = common_host_bytes,
                                   .cumulative = common_host_bytes,
                                   .budget = common_host_bytes});
  return PreparedKernelPipeline{
      .owner = std::static_pointer_cast<void>(pipeline),
      .ok = true,
      .reason = "ok",
  };
}

rund::AccelCheck
SeedPreparedKernelPipelineGeneration(const PreparedKernelPipeline &prepared,
                                     const std::uint32_t generation) noexcept {
  auto *const pipeline =
      static_cast<prepared::PipelineState *>(prepared.owner.get());
  if (!prepared.ok || pipeline == nullptr || pipeline->ops == nullptr ||
      pipeline->backend == nullptr ||
      pipeline->ops->seed_prepared_pipeline_generation == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  return pipeline->ops->seed_prepared_pipeline_generation(pipeline->backend,
                                                          generation);
}

PreparedPipelineMemory ReadPreparedKernelPipelineMemory(
    const PreparedKernelPipeline &prepared) noexcept {
  const auto *const pipeline =
      static_cast<const prepared::PipelineState *>(prepared.owner.get());
  return prepared.ok && pipeline != nullptr ? pipeline->memory
                                            : PreparedPipelineMemory{};
}

PreparedPipelineStatusLayout ReadPreparedKernelPipelineStatus(
    const PreparedKernelPipeline &prepared) noexcept {
  const auto *const pipeline =
      static_cast<const prepared::PipelineState *>(prepared.owner.get());
  return prepared.ok && pipeline != nullptr ? pipeline->status
                                            : PreparedPipelineStatusLayout{};
}

} // namespace rund::node::accel::detail
