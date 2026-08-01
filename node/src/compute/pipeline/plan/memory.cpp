#include "arena.hpp"
#include "compare.hpp"
#include "local.hpp"
#include "prepare.hpp"

#include "../../buffer/local.hpp"
#include "../../job/local.hpp"
#include "../../memory/arena.hpp"
#include "../../type.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/abi/observe.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rund::compute::detail {

[[nodiscard]] Result<std::shared_ptr<const PipelineMemoryPlan>>
plan_memory(const PipelineBuildState &build) {
  try {
    auto result = std::make_shared<PipelineMemoryPlan>();
    PipelinePlan &summary = result->summary;
    std::unordered_set<const ProgramState *> programs;
    programs.reserve(build.steps.size());
    for (const PipelineBuildStep &step : build.steps) {
      if (step.program == nullptr) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineInvalid);
      }
      if (programs.insert(step.program.get()).second &&
          !kernel::checked::add(
              summary.node_count,
              static_cast<std::uint64_t>(step.program->graph_info.nodes.size()),
              summary.node_count)) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
      }
    }
    std::uint64_t nested_commands = 0u;
    std::uint64_t nested_templates = 0u;
    std::uint64_t logical_workspace = 0u;
    std::uint64_t live_workspace = 0u;
    std::size_t covered_until = 0u;
    for (std::size_t nested_index = 0u;
         nested_index < build.nested_windows.size(); ++nested_index) {
      const PipelineBuildNestedWindow &nested =
          build.nested_windows[nested_index];
      const std::size_t step_count = build.steps.size();
      const bool seed_range =
          nested.seed_first < step_count && nested.seed_count != 0u &&
          nested.seed_count <= step_count - nested.seed_first;
      const std::size_t expected_action_first =
          seed_range ? nested.seed_first + nested.seed_count : step_count;
      const bool action_range =
          seed_range && nested.action_first == expected_action_first &&
          nested.action_count <= step_count - nested.action_first;
      const std::size_t expected_fold_first =
          action_range ? nested.action_first + nested.action_count : step_count;
      const bool fold_range = action_range &&
                              nested.fold_first == expected_fold_first &&
                              nested.fold_first < step_count &&
                              3u <= step_count - nested.fold_first;
      const std::size_t expected_end =
          fold_range ? nested.fold_first + 3u : step_count;
      const bool window_shape =
          nested.maximum != 0u && nested.tile != 0u &&
          nested.tile <= nested.maximum &&
          nested.seed_count == nested.maximum / nested.tile +
                                   (nested.maximum % nested.tile != 0u);
      if (nested_index >= std::numeric_limits<std::uint16_t>::max() ||
          nested.begin < covered_until || nested.begin != nested.seed_first ||
          nested.begin >= nested.end || !fold_range ||
          nested.end != expected_end || !window_shape) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineInvalid);
      }
      const auto nested_id = static_cast<std::uint16_t>(nested_index + 1u);
      const auto valid_phase = [&](const std::size_t first,
                                   const std::size_t count,
                                   const PipelineRoute route) {
        if (count == 0u) {
          return true;
        }
        const ProgramState *const program = build.steps[first].program.get();
        for (std::size_t offset = 0u; offset < count; ++offset) {
          const PipelineBuildStep &step = build.steps[first + offset];
          if (step.program.get() != program || step.nested != nested_id ||
              step.route != route || step.iteration != offset ||
              step.iteration_bound != count) {
            return false;
          }
        }
        return true;
      };
      if (!valid_phase(nested.seed_first, nested.seed_count,
                       PipelineRoute::NestedSeed) ||
          !valid_phase(nested.action_first, nested.action_count,
                       PipelineRoute::NestedAction) ||
          !valid_phase(nested.fold_first, 3u, PipelineRoute::NestedFold)) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineInvalid);
      }
      covered_until = nested.end;
      summary.outer_window_count =
          std::max(summary.outer_window_count,
                   static_cast<std::uint64_t>(nested.seed_count));
      summary.tile_capacity = std::max(summary.tile_capacity,
                                       static_cast<std::uint64_t>(nested.tile));
      summary.inner_iteration_count =
          std::max(summary.inner_iteration_count,
                   static_cast<std::uint64_t>(nested.action_count));
      std::uint64_t commands_per_window = 0u;
      std::uint64_t commands = 0u;
      if (!kernel::checked::add(static_cast<std::uint64_t>(nested.action_count),
                                2u, commands_per_window) ||
          !kernel::checked::mul(static_cast<std::uint64_t>(nested.seed_count),
                                commands_per_window, commands) ||
          !kernel::checked::add(nested_commands, commands, nested_commands)) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
      }
      std::uint64_t templates = 0u;
      if (!kernel::checked::add(static_cast<std::uint64_t>(nested.seed_count),
                                static_cast<std::uint64_t>(nested.action_count),
                                templates) ||
          !kernel::checked::add(templates, 3u, templates) ||
          !kernel::checked::add(nested_templates, templates,
                                nested_templates)) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
      }
      const graph::MemoryPlan &seed_memory =
          build.steps[nested.seed_first].program->graph_info.memory;
      const graph::MemoryPlan &fold_memory =
          build.steps[nested.fold_first].program->graph_info.memory;
      const std::uint64_t action_logical =
          nested.action_count == 0u
              ? 0u
              : build.steps[nested.action_first]
                    .program->graph_info.memory.logical_bytes;
      std::uint64_t per_window = 0u;
      std::uint64_t action_total = 0u;
      std::uint64_t all_windows = 0u;
      if (!kernel::checked::mul(action_logical,
                                static_cast<std::uint64_t>(nested.action_count),
                                action_total) ||
          !kernel::checked::add(seed_memory.logical_bytes, action_total,
                                per_window) ||
          !kernel::checked::add(per_window, fold_memory.logical_bytes,
                                per_window) ||
          !kernel::checked::mul(per_window,
                                static_cast<std::uint64_t>(nested.seed_count),
                                all_windows) ||
          !kernel::checked::add(logical_workspace, all_windows,
                                logical_workspace)) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
      }
    }
    std::size_t nested_index = 0u;
    for (std::size_t step_index = 0u; step_index < build.steps.size();
         ++step_index) {
      while (nested_index < build.nested_windows.size() &&
             step_index >= build.nested_windows[nested_index].end) {
        ++nested_index;
      }
      const bool nested =
          nested_index < build.nested_windows.size() &&
          step_index >= build.nested_windows[nested_index].begin;
      const PipelineBuildStep &step = build.steps[step_index];
      if (!nested) {
        if (step.nested != 0u || step.route != PipelineRoute::Ordinary) {
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              Reason::PipelineInvalid);
        }
        if (!kernel::checked::add(logical_workspace,
                                  step.program->graph_info.memory.logical_bytes,
                                  logical_workspace)) {
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              Reason::PipelineCapacity);
        }
      }
      live_workspace =
          std::max(live_workspace, step.program->graph_info.memory.live_bytes);
    }
    std::unordered_map<const BufferState *, std::uint8_t> external;
    external.reserve(std::min(build.binding_count, PipelineResourceCapacity));
    const auto admit = [&](const PipelineBinding &binding) {
      if (binding.owner != PipelineBinding::external) {
        return true;
      }
      if (binding.buffer == nullptr) {
        return false;
      }
      const auto [position, inserted] =
          external.emplace(binding.buffer.get(), 0u);
      return !inserted || kernel::checked::add(summary.persistent_bytes,
                                               binding.buffer->bytes,
                                               summary.persistent_bytes);
    };
    for (const PipelineBuildStep &step : build.steps) {
      for (const PipelineBinding &binding : step.inputs) {
        if (!admit(binding)) {
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              Reason::PipelineCapacity);
        }
      }
      for (const PipelineBinding &binding : step.outputs) {
        if (!admit(binding)) {
          return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
              Reason::PipelineCapacity);
        }
      }
    }
    for (const PipelineBuildStatePair &pair : build.state_pairs) {
      if (!admit(pair.published) || !admit(pair.pending)) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
      }
    }
    for (const PipelineBuildPublish &publication : build.publications) {
      if (!admit(publication.source) || !admit(publication.target) ||
          (publication.kind == PipelinePublishKind::Window &&
           !admit(publication.count))) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
      }
      std::uint64_t bytes = 0u;
      const std::uint64_t elements =
          publication.kind == PipelinePublishKind::Window
              ? publication.maximum
              : publication.source.count;
      const std::uint64_t occurrences =
          publication.kind == PipelinePublishKind::Window
              ? (publication.maximum / publication.tile +
                 (publication.maximum % publication.tile == 0u ? 0u : 1u))
              : 1u;
      if (publication.source.element_bytes == 0u ||
          !kernel::checked::mul(elements, publication.source.element_bytes,
                                bytes) ||
          !kernel::checked::add(summary.publish_bytes, bytes,
                                summary.publish_bytes) ||
          !kernel::checked::add(summary.publish_count, occurrences,
                                summary.publish_count)) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
      }
    }

    for (const PipelineInternal &internal : build.internals) {
      const std::size_t width = type_bytes(internal.type);
      std::uint64_t bytes = 0u;
      if (width == 0u ||
          !kernel::checked::mul(static_cast<std::uint64_t>(internal.count),
                                width, bytes) ||
          !kernel::checked::add(summary.state_bytes, bytes,
                                summary.state_bytes)) {
        return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
            Reason::PipelineCapacity);
      }
    }
    const Status schedule = plan_pipeline_schedule(build, *result);
    if (!schedule) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          schedule.reason());
    }

    const Status arena =
        plan_pipeline_arena(*build.device, build.steps, *result);
    if (!arena) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          arena.reason());
    }
    const Status views =
        plan_pipeline_views(*build.device, build.steps, *result);
    if (!views) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          views.reason());
    }
    const Status scratch =
        plan_pipeline_scratch(*build.device, build.steps, *result);
    if (!scratch) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          scratch.reason());
    }
    summary.allocation_count = build.internals.size() + result->chunks.size() +
                               result->view_chunks.size() +
                               result->scratch_chunks.size();
    std::uint64_t infrastructure_bytes = 0u;
    if (!kernel::checked::add(summary.state_bytes, summary.prepared_bytes,
                              infrastructure_bytes) ||
        !kernel::checked::add(infrastructure_bytes, summary.transient_bytes,
                              summary.peak_bytes) ||
        !kernel::checked::add(summary.persistent_bytes, summary.peak_bytes,
                              summary.total_bytes)) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          Reason::PipelineCapacity);
    }
    std::uint64_t flat_templates = 0u;
    std::uint64_t flat_commands = 0u;
    for (std::size_t index = 0u; index < build.steps.size(); ++index) {
      const PipelineBuildStep &step = build.steps[index];
      if (step.route != PipelineRoute::Ordinary) {
        continue;
      }
      ++flat_commands;
      const std::uint32_t reusable_from = 3u;
      const bool reused = step.iteration_bound > 1u &&
                          step.iteration >= reusable_from && index >= 2u &&
                          same_recurrence_phase(step, build.steps[index - 2u]);
      if (!reused) {
        ++flat_templates;
      }
    }
    if (!kernel::checked::add(flat_templates, nested_templates,
                              summary.prepared_template_count) ||
        !kernel::checked::add(flat_commands, nested_commands,
                              summary.prepared_command_count)) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          Reason::PipelineCapacity);
    }
    if (!kernel::checked::add(infrastructure_bytes, logical_workspace,
                              summary.logical_bytes) ||
        !kernel::checked::add(infrastructure_bytes, live_workspace,
                              summary.live_bytes) ||
        !kernel::checked::add(infrastructure_bytes, summary.transient_bytes,
                              summary.physical_bytes)) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          Reason::PipelineCapacity);
    }
    if (summary.physical_bytes != summary.peak_bytes) {
      return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
          Reason::PipelineInvalid);
    }
    return Result<std::shared_ptr<const PipelineMemoryPlan>>::success(
        std::move(result));
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<const PipelineMemoryPlan>>::fail(
        Reason::PipelineCapacity);
  }
}

[[nodiscard]] Status materialize_pipeline(PipelineBuildState &build) {
  try {
    std::vector<std::shared_ptr<BufferState>> owners;
    owners.reserve(build.internals.size());
    std::array<std::uint32_t, PipelineIterationCapacity> ordinals{};
    for (std::size_t index = 0u; index < build.internals.size(); ++index) {
      const PipelineInternal &internal = build.internals[index];
      auto made = make_input_binding_buffer(build.device, internal.type,
                                            internal.count);
      if (!made) {
        return Status::fail(made.reason());
      }
      if (internal.fill == PipelineFill::Ordinal) {
        if (internal.type != Type::U32 || internal.count > ordinals.size()) {
          return Status::fail(Reason::PipelineInvalid);
        }
        std::iota(ordinals.begin(), ordinals.begin() + internal.count, 0u);
        WriteStats writes{};
        const Status written = write_buffer(*made,
                                            HostView{.data = ordinals.data(),
                                                     .count = internal.count,
                                                     .type = Type::U32},
                                            writes);
        if (!written) {
          return written;
        }
      }
      owners.push_back(std::move(made).value());
    }
    const auto resolve = [&](PipelineBinding &binding) {
      if (binding.owner == PipelineBinding::external) {
        return binding.buffer != nullptr;
      }
      if (binding.owner >= owners.size() || binding.buffer != nullptr) {
        return false;
      }
      binding.buffer = owners[binding.owner];
      return true;
    };
    for (PipelineBuildStep &step : build.steps) {
      for (PipelineBinding &binding : step.inputs) {
        if (!resolve(binding)) {
          return Status::fail(Reason::PipelineInvalid);
        }
      }
      for (PipelineBinding &binding : step.outputs) {
        if (!resolve(binding)) {
          return Status::fail(Reason::PipelineInvalid);
        }
      }
    }
    for (PipelineBuildPublish &publication : build.publications) {
      if (!resolve(publication.source) || !resolve(publication.target) ||
          (publication.kind == PipelinePublishKind::Window &&
           !resolve(publication.count))) {
        return Status::fail(Reason::PipelineInvalid);
      }
    }
    return Status::success();
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::PipelineCapacity);
  }
}

[[nodiscard]] Result<PipelineMemorySet>
make_pipeline_memory(const std::shared_ptr<DeviceState> &device,
                     const std::span<const PipelineBuildStep> steps,
                     const PipelineMemoryPlan &plan) {
  try {
    PipelineMemorySet result;
    result.steps.resize(steps.size());
    if (plan.steps.size() != steps.size() + 1u || plan.steps.empty() ||
        plan.steps.front() != 0u || plan.steps.back() != plan.offsets.size() ||
        plan.owners.size() != plan.offsets.size()) {
      return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
    }
    result.buffers.reserve(plan.chunks.size());
    for (const std::size_t count : plan.chunks) {
      if (count == 0u) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      auto made = make_workspace_buffer(device, count);
      if (!made) {
        return Result<PipelineMemorySet>::fail(made.reason());
      }
      result.buffers.push_back(std::move(made).value());
    }
    if (plan.views.size() != steps.size()) {
      return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
    }
    result.prepared.reserve(plan.view_chunks.size() +
                            plan.scratch_chunks.size());
    if (!plan.view_chunks.empty() || !plan.scratch_chunks.empty()) {
      result.arena = std::make_shared<JobArena>();
      result.arena->buffers.reserve(plan.view_chunks.size() +
                                    plan.scratch_chunks.size());
      result.arena->slots.reserve(plan.view_slots.size() +
                                  plan.scratch_chunks.size());
    }
    for (const std::size_t words : plan.view_chunks) {
      if (words == 0u) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      auto made = make_workspace_buffer(device, words);
      if (!made) {
        return Result<PipelineMemorySet>::fail(made.reason());
      }
      result.prepared.push_back(std::move(made).value());
      result.arena->buffers.push_back(result.prepared.back());
    }
    for (const PipelineMemoryPlan::ViewSlot slot : plan.view_slots) {
      if (slot.words == 0u || slot.owner >= plan.view_chunks.size() ||
          slot.offset_words > plan.view_chunks[slot.owner] ||
          slot.words > plan.view_chunks[slot.owner] - slot.offset_words) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      result.arena->slots.push_back(JobArenaSlot{
          .words = slot.words,
          .owner = slot.owner,
          .offset_words = slot.offset_words,
      });
    }
    for (const std::size_t words : plan.scratch_chunks) {
      if (words == 0u || result.arena == nullptr) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      auto made = make_workspace_buffer(device, words);
      if (!made) {
        return Result<PipelineMemorySet>::fail(made.reason());
      }
      result.prepared.push_back(std::move(made).value());
      result.arena->buffers.push_back(result.prepared.back());
      const std::size_t slot = result.arena->slots.size();
      result.arena->slots.push_back(
          JobArenaSlot{.words = words,
                       .owner = result.arena->buffers.size() - 1u,
                       .offset_words = 0u});
      std::uint64_t bytes = 0u;
      if (!kernel::checked::mul(static_cast<std::uint64_t>(words), memory::Word,
                                bytes)) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineCapacity);
      }
      result.arena->scratch.push_back(
          node::accel::detail::KernelScratchPage{.slot = slot, .bytes = bytes});
    }
    for (std::size_t step_index = 0u; step_index < steps.size(); ++step_index) {
      const PipelineBuildStep &step = steps[step_index];
      const auto &chunks = step.program->chunks;
      if (step.iteration_bound > 1u && step.iteration != 0u) {
        if (step_index == 0u || result.steps[step_index - 1u] == nullptr ||
            result.steps[step_index - 1u]->program != step.program) {
          return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
        }
        result.steps[step_index] = result.steps[step_index - 1u];
        continue;
      }
      if (chunks.empty() && plan.views[step_index].empty() &&
          step.iteration_bound == 1u && result.arena == nullptr) {
        continue;
      }
      const std::vector<std::uint32_t> &order = step.program->chunk_order;
      if (order.size() != chunks.size()) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      auto workspace = std::make_shared<JobWorkspace>();
      workspace->program = step.program;
      workspace->arena = result.arena;
      workspace->buffers.resize(chunks.size());
      workspace->offsets.resize(chunks.size());
      const std::size_t begin = plan.steps[step_index];
      const std::size_t end = plan.steps[step_index + 1u];
      if (begin > end || end > plan.offsets.size() ||
          end - begin != order.size()) {
        return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
      }
      for (std::size_t rank = 0u; rank < order.size(); ++rank) {
        const std::size_t chunk = order[rank];
        const std::size_t owner = plan.owners[begin + rank];
        const std::size_t offset = plan.offsets[begin + rank];
        if (chunk >= chunks.size() || owner >= result.buffers.size() ||
            owner >= plan.chunks.size() || offset > plan.chunks[owner] ||
            chunks[chunk].count > plan.chunks[owner] - offset) {
          return Result<PipelineMemorySet>::fail(Reason::PipelineInvalid);
        }
        workspace->buffers[chunk] = result.buffers[owner];
        workspace->offsets[chunk] = offset;
      }
      result.steps[step_index] = std::move(workspace);
    }
    return Result<PipelineMemorySet>::success(std::move(result));
  } catch (const std::bad_alloc &) {
    return Result<PipelineMemorySet>::fail(Reason::BufferCapacity);
  }
}

} // namespace rund::compute::detail
