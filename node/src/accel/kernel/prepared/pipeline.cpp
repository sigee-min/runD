#include "../prepared.hpp"

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
      first_window->outer_bound == 0u || first_window->inner_bound == 0u) {
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

[[nodiscard]] bool
expand_pipeline(const std::span<const BackendBatchEntry> templates,
                const std::span<const std::uint8_t> template_barriers,
                ExpandedPipeline &expanded) {
  if (templates.empty() || templates.size() != template_barriers.size()) {
    return false;
  }
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
    const std::uint64_t commands_per_outer =
        static_cast<std::uint64_t>(inner_bound) + 2u;
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

  const std::size_t capacity = static_cast<std::size_t>(command_count);
  expanded.commands.reserve(capacity);
  expanded.barriers.reserve(capacity);
  expanded.windows.reserve(capacity);
  const auto append = [&](const std::size_t template_index,
                          const std::uint8_t barrier, const std::uint32_t outer,
                          const std::uint32_t outer_bound,
                          const std::uint32_t inner,
                          const std::uint32_t inner_bound,
                          const std::uint32_t route) {
    if (template_index >= templates.size() ||
        expanded.commands.size() >= std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    BackendBatchEntry command = templates[template_index];
    command.template_index = static_cast<std::uint32_t>(template_index);
    command.occurrence_index =
        static_cast<std::uint32_t>(expanded.commands.size());
    if (command.recurrence.window != nullptr) {
      expanded.windows.push_back(*command.recurrence.window);
      BackendWindow &window = expanded.windows.back();
      window.iteration = outer;
      window.bound = outer_bound;
      window.outer_iteration = outer;
      window.outer_bound = outer_bound;
      window.inner_iteration = inner;
      window.inner_bound = inner_bound;
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
    for (std::uint32_t outer = 0u; outer < outer_bound; ++outer) {
      const bool first_command = expanded.commands.empty();
      const std::uint8_t seed_barrier =
          first_command ? template_barriers[index + outer] : 1u;
      if (!append(index + outer, seed_barrier, outer, outer_bound, 0u,
                  inner_bound, 0u)) {
        return false;
      }
      for (std::uint32_t inner = 0u; inner < inner_bound; ++inner) {
        if (!append(action_first + inner, 1u, outer, outer_bound, inner,
                    inner_bound, 0u)) {
          return false;
        }
      }
      const std::uint32_t route =
          outer == 0u ? 0u : ((outer & 1u) != 0u ? 1u : 2u);
      if (!append(fold_first + route, 1u, outer, outer_bound, inner_bound,
                  inner_bound, route)) {
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
    if (publication.target_handle == nullptr ||
        publication.state >= state_count ||
        publication.final >= publication.sources.size() ||
        target.stride_bytes < target.element_bytes ||
        target.usage != rund::kernel::kResidentUsageWrite) {
      return PreparedKernelPipeline{.reason = invalid.reason};
    }
    for (const BackendRead &read : publication.sources) {
      const auto &source = read.source;
      if (read.handle == nullptr || source.count != target.count ||
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
    if (!expand_pipeline(templates, barriers, expanded)) {
      return PreparedKernelPipeline{.reason = invalid.reason};
    }
  } catch (const std::bad_alloc &) {
    return PreparedKernelPipeline{.reason = "compute_pipeline_capacity"};
  }
  if (expanded.commands.size() > std::numeric_limits<std::uint32_t>::max() ||
      !PreparePipelineStatusLayout(
          pipeline->status, declared_steps, declared_step_count,
          static_cast<std::uint32_t>(expanded.commands.size()),
          generation_stride)) {
    return PreparedKernelPipeline{.reason = invalid.reason};
  }
  for (const BackendBatchEntry &command : expanded.commands) {
    if (command.template_index >= pipeline->size ||
        pipeline->states[command.template_index] == nullptr) {
      return PreparedKernelPipeline{.reason = invalid.reason};
    }
    prepared::Accumulate(pipeline->counts,
                         *pipeline->states[command.template_index]);
  }
  const rund::AccelCheck built = pipeline->ops->prepare_pipeline(
      templates, expanded.commands, expanded.barriers, publications,
      pipeline->status, profile_steps, pipeline->backend, pipeline->memory);
  if (!built.ok) {
    return PreparedKernelPipeline{.reason = built.reason};
  }
  if (!ValidPreparedPipelineStatusLayout(
          pipeline->status, declared_steps, declared_step_count,
          static_cast<std::uint32_t>(expanded.commands.size()),
          generation_stride)) {
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
