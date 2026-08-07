#include "prepare.hpp"

#include "../../job/local.hpp"
#include "../claim.hpp"
#include "../local.hpp"
#include "../state.hpp"
#include "compare.hpp"
#include "local.hpp"
#include "resource.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace rund::compute::detail {

namespace {

[[nodiscard]] bool
job_binding_matches(const PipelineState &state,
                    const PipelinePublicationViewPlan &planned,
                    const std::shared_ptr<BufferState> &owner,
                    const JobBufferView &view, const bool alternate) noexcept {
  const PipelinePublicationViewIdentity &identity = planned.identity;
  const PipelineResource *const resource =
      selected_pipeline_resource(state, identity.resource_ordinal, alternate);
  return resource != nullptr && resource->buffer != nullptr &&
         owner == resource->buffer && resource->type == planned.type &&
         resource->format == planned.format &&
         resource->bytes == identity.backing_bytes &&
         view.count == identity.count &&
         view.element_bytes == identity.element_bytes &&
         view.offset == planned.offset && view.stride == planned.stride;
}

[[nodiscard]] bool job_step_binding_matches(
    const PipelineState &state, const std::size_t step_index, const bool input,
    const std::uint32_t slot, const PipelinePublicationViewPlan &planned,
    const bool alternate) noexcept {
  if (step_index >= state.steps.size()) {
    return false;
  }
  const std::shared_ptr<JobState> &job =
      alternate ? state.steps[step_index].alternate_job
                : state.steps[step_index].job;
  if (job == nullptr) {
    return false;
  }
  const auto &owners = input ? job->inputs : job->outputs;
  const auto &views = input ? job->input_views : job->output_views;
  return owners.size() == views.size() && slot < owners.size() &&
         job_binding_matches(state, planned, owners[slot], views[slot],
                             alternate);
}

[[nodiscard]] bool job_count_matches(const PipelineState &state,
                                     const std::size_t step_index,
                                     const PipelinePublicationViewPlan &count,
                                     const std::uint32_t count_input,
                                     const bool alternate) noexcept {
  if (step_index >= state.steps.size()) {
    return false;
  }
  const std::shared_ptr<JobState> &job =
      alternate ? state.steps[step_index].alternate_job
                : state.steps[step_index].job;
  return job != nullptr && count_input < job->inputs.size() &&
         job_step_binding_matches(state, step_index, true, count_input, count,
                                  alternate);
}

[[nodiscard]] bool
terminal_step_bindings_match(const PipelineState &state,
                             const PipelineWindow &window,
                             const PipelineTerminalPublicationPlan &terminal,
                             const bool alternate) noexcept {
  if (terminal.output.value >= window.recurrent_output_count) {
    return false;
  }
  if (window.nested()) {
    const std::size_t fold_first = window.nested_shape.fold_first();
    constexpr std::array input_banks{
        PipelineWindow::seed, PipelineWindow::first, PipelineWindow::second};
    constexpr std::array output_banks{
        PipelineWindow::first, PipelineWindow::second, PipelineWindow::first};
    const std::size_t fold_count = window.nested_shape.fold_count();
    if (!window.nested_shape.valid() || fold_count != input_banks.size() ||
        fold_count != output_banks.size() || fold_first > state.steps.size() ||
        state.steps.size() - fold_first < fold_count) {
      return false;
    }
    for (std::size_t route = 0u; route < fold_count; ++route) {
      const std::size_t step_index = fold_first + route;
      if (!job_step_binding_matches(
              state, step_index, true, terminal.output.value,
              terminal.sources[input_banks[route]], alternate) ||
          !job_step_binding_matches(
              state, step_index, false, terminal.output.value,
              terminal.sources[output_banks[route]], alternate)) {
        return false;
      }
    }
    return true;
  }

  bool saw_step = false;
  const std::uint16_t window_index =
      static_cast<std::uint16_t>(terminal.state + 1u);
  for (std::size_t step_index = 0u; step_index < state.steps.size();
       ++step_index) {
    const PipelineStep &step = state.steps[step_index];
    if (step.window != window_index || step.route != PipelineRoute::Ordinary) {
      continue;
    }
    saw_step = true;
    const std::uint32_t input_bank =
        step.iteration == 0u
            ? PipelineWindow::seed
            : ((step.iteration & 1u) != 0u ? PipelineWindow::first
                                           : PipelineWindow::second);
    const std::uint32_t output_bank = (step.iteration & 1u) == 0u
                                          ? PipelineWindow::first
                                          : PipelineWindow::second;
    if (!job_step_binding_matches(state, step_index, true,
                                  terminal.output.value,
                                  terminal.sources[input_bank], alternate) ||
        !job_step_binding_matches(state, step_index, false,
                                  terminal.output.value,
                                  terminal.sources[output_bank], alternate)) {
      return false;
    }
  }
  return saw_step;
}

[[nodiscard]] Status
validate_publication_job_bindings(const PipelineState &state) noexcept {
  for (std::size_t state_index = 0u; state_index < state.windows.size();
       ++state_index) {
    const PipelineWindow &window = state.windows[state_index];
    const std::uint16_t window_index =
        static_cast<std::uint16_t>(state_index + 1u);
    const std::size_t count_step =
        window.nested() ? window.nested_shape.seed_first() : window.first_step;
    if (count_step >= state.steps.size() ||
        state.steps[count_step].window != window_index) {
      return Status::fail(Reason::PipelineInvalid);
    }
    for (const bool alternate : {false, true}) {
      if (alternate && !state.transactional) {
        continue;
      }
      bool saw_count = false;
      for (std::size_t step_index = 0u; step_index < state.steps.size();
           ++step_index) {
        const PipelineStep &step = state.steps[step_index];
        const bool consumes_count =
            step.window == window_index &&
            (!window.nested() || step.route == PipelineRoute::NestedSeed);
        if (!consumes_count) {
          continue;
        }
        saw_count = true;
        if (!job_count_matches(state, step_index, window.control.count,
                               window.control.count_input, alternate)) {
          return Status::fail(Reason::PipelineInvalid);
        }
      }
      if (!saw_count) {
        return Status::fail(Reason::PipelineInvalid);
      }
    }
  }

  std::array<std::uint32_t, PipelineStepCapacity> terminal_counts{};
  if (state.windows.size() > terminal_counts.size()) {
    return Status::fail(Reason::PipelineCapacity);
  }
  for (const PipelinePublicationPlan &publication : state.publications) {
    if (const auto *terminal =
            std::get_if<PipelineTerminalPublicationPlan>(&publication)) {
      if (terminal->state >= state.windows.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      ++terminal_counts[terminal->state];
      const PipelineWindow &window = state.windows[terminal->state];
      for (const bool alternate : {false, true}) {
        if (alternate && !state.transactional) {
          continue;
        }
        if (!terminal_step_bindings_match(state, window, *terminal,
                                          alternate)) {
          return Status::fail(Reason::PipelineInvalid);
        }
      }
      continue;
    }
    const auto &window_publication =
        std::get<PipelineWindowPublicationPlan>(publication);
    if (window_publication.state >= state.windows.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const PipelineWindow &window = state.windows[window_publication.state];
    for (const bool alternate : {false, true}) {
      if (alternate && !state.transactional) {
        continue;
      }
      const std::size_t fold_first = window.nested_shape.fold_first();
      const std::size_t fold_count = window.nested_shape.fold_count();
      if (!window.nested() || !window.nested_shape.valid() ||
          fold_first > state.steps.size() ||
          state.steps.size() - fold_first < fold_count) {
        return Status::fail(Reason::PipelineInvalid);
      }
      for (std::size_t route = 0u; route < fold_count; ++route) {
        if (!job_step_binding_matches(state, fold_first + route, false,
                                      window_publication.output.value,
                                      window_publication.source, alternate)) {
          return Status::fail(Reason::PipelineInvalid);
        }
      }
    }
  }
  for (std::size_t state_index = 0u; state_index < state.windows.size();
       ++state_index) {
    if (terminal_counts[state_index] !=
        state.windows[state_index].recurrent_output_count) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  return Status::success();
}

} // namespace

Status bind_pipeline(const std::shared_ptr<PipelineBuildState> &build,
                     PipelinePrepare &prepare) {
  std::shared_ptr<PipelineState> &state = prepare.state;
  const std::uint64_t status_entry_count = prepare.status_count;
  if (state == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  // Phase 3 may allocate and prepare private Program entries only after the
  // complete public Pipeline contract has passed its cold validation.
  const Status resources_valid = validate_pipeline_resources(*state);
  if (!resources_valid) {
    return Status::fail(resources_valid.reason());
  }
  if (build->memory == nullptr || build->memory->frozen == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const PipelineBuildSnapshot &frozen = *build->memory->frozen;
  if (build->memory->step_resources.size() != frozen.steps.size() ||
      build->materialized_resources.size() != build->memory->resources.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  state->accel_templates = {};
  if (state->device->backend != Backend::Cpu) {
    if (!build->memory->accel_preparation.ok) {
      return Status::fail(Reason::PipelineInvalid);
    }
    state->accel_templates.limit = build->memory->accel_preparation;
  }
  auto memory =
      make_pipeline_memory(state->device, frozen.steps, *build->memory);
  if (!memory) {
    return Status::fail(memory.reason());
  }
  state->prepared_buffers = std::move(memory->prepared);
  state->cpu_prepared_arena = std::move(memory->cpu_prepared_arena);
  state->cpu_storage = std::move(memory->cpu_storage);
  if (build->memory->job_owners.size() != state->steps.size() ||
      memory->cpu_storage_by_step.size() != state->steps.size() ||
      build->memory->cpu_view_layouts.size() != state->steps.size() ||
      build->memory->cpu_route_slices.size() != state->steps.size() ||
      (state->device->backend == Backend::Cpu &&
       build->memory->cpu_job_slices.size() != state->steps.size()) ||
      (state->transactional && state->device->backend == Backend::Cpu &&
       build->memory->cpu_alternate_job_slices.size() != state->steps.size()) ||
      (state->transactional
           ? build->memory->cpu_alternate_route_slices.size() !=
                 state->steps.size()
           : !build->memory->cpu_alternate_route_slices.empty())) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const auto alternate_owner =
      [&](const std::uint32_t ordinal) -> const std::shared_ptr<BufferState> & {
    const PipelineResource *const selected =
        selected_pipeline_resource(*state, ordinal, true);
    static const std::shared_ptr<BufferState> invalid{};
    return selected == nullptr ? invalid : selected->buffer;
  };
  const auto primary_owner =
      [&](const std::uint32_t ordinal) -> const std::shared_ptr<BufferState> & {
    static const std::shared_ptr<BufferState> invalid{};
    return ordinal < build->materialized_resources.size()
               ? build->materialized_resources[ordinal]
               : invalid;
  };
  for (std::size_t step_index = 0u; step_index < state->steps.size();
       ++step_index) {
    PipelineStep &step = state->steps[step_index];
    const PipelineFrozenStep &declared = frozen.steps[step_index];
    const PipelineStepResourcePlan &sealed =
        build->memory->step_resources[step_index];
    const std::shared_ptr<JobWorkspace> &workspace = memory->steps[step_index];
    const std::size_t job_owner = build->memory->job_owners[step_index];
    if (job_owner != step_index) {
      if (job_owner >= step_index || state->steps[job_owner].job == nullptr ||
          build->memory->step_resources[job_owner] != sealed ||
          (state->transactional &&
           state->steps[job_owner].alternate_job == nullptr)) {
        return Status::fail(Reason::PipelineInvalid);
      }
      step.job = state->steps[job_owner].job;
      step.alternate_job = state->steps[job_owner].alternate_job;
      continue;
    }
    if (state->device->backend == Backend::Cpu) {
      std::shared_ptr<CpuGraphStorage> cpu_storage;
      const CpuRunRoutePlan *cpu_route = nullptr;
      const CpuRunRouteSlice *cpu_route_slice = nullptr;
      const std::size_t cpu_owner = memory->cpu_storage_by_step[step_index];
      if (cpu_owner != std::numeric_limits<std::size_t>::max()) {
        if (cpu_owner >= state->cpu_storage.size() ||
            cpu_owner >= build->memory->cpu_route_plans.size() ||
            state->cpu_prepared_arena == nullptr) {
          return Status::fail(Reason::PipelineInvalid);
        }
        cpu_storage = state->cpu_storage[cpu_owner];
        cpu_route = &build->memory->cpu_route_plans[cpu_owner];
        cpu_route_slice = &build->memory->cpu_route_slices[step_index];
      }
      const CpuViewTransferLayout &cpu_views =
          build->memory->cpu_view_layouts[step_index];
      if (state->cpu_prepared_arena == nullptr) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const auto prepare_cpu_job =
          [&](const bool alternate, const CpuJobBindingSlice &binding_slice,
              const CpuRunRouteSlice *const route_slice)
          -> Result<std::shared_ptr<JobState>> {
        CpuJobBindingStorage bindings{};
        if (!state->cpu_prepared_arena->view(binding_slice, bindings) ||
            bindings.inputs.size() != sealed.inputs.size() ||
            bindings.outputs.size() != sealed.physical_sources.size() ||
            bindings.input_views.size() != sealed.inputs.size() ||
            bindings.output_views.size() != sealed.physical_sources.size() ||
            bindings.kernel_views.size() !=
                build->memory->views[step_index].size()) {
          return Result<std::shared_ptr<JobState>>::fail(
              Reason::PipelineInvalid);
        }
        for (std::size_t index = 0u; index < sealed.inputs.size(); ++index) {
          const PipelineResolvedViewPlan &view = sealed.inputs[index];
          bindings.inputs[index] = alternate ? alternate_owner(view.resource)
                                             : primary_owner(view.resource);
          bindings.input_views[index] =
              PipelineScheduleResources::job_view(view);
        }
        for (std::size_t index = 0u; index < sealed.physical_sources.size();
             ++index) {
          const std::uint32_t source = sealed.physical_sources[index];
          if (source >= sealed.outputs.size()) {
            return Result<std::shared_ptr<JobState>>::fail(
                Reason::PipelineInvalid);
          }
          const PipelineResolvedViewPlan &view = sealed.outputs[source].view;
          bindings.outputs[index] = alternate ? alternate_owner(view.resource)
                                              : primary_owner(view.resource);
          bindings.output_views[index] =
              PipelineScheduleResources::job_view(view);
        }
        std::copy(build->memory->views[step_index].begin(),
                  build->memory->views[step_index].end(),
                  bindings.kernel_views.begin());
        return prepare_pipeline_cpu_job(
            step.program, bindings, state->cpu_prepared_arena, workspace,
            cpu_storage, cpu_route, route_slice, cpu_views);
      };
      auto job = prepare_cpu_job(
          false, build->memory->cpu_job_slices[step_index], cpu_route_slice);
      if (!job) {
        prepare.failure = job.location();
        prepare.failure.step = declared.logical_step;
        prepare.failure.iteration = declared.iteration;
        return Status::fail(job.reason());
      }
      step.job = std::move(job).value();
      if (state->transactional) {
        const CpuRunRouteSlice *const alternate_route_slice =
            cpu_route == nullptr
                ? nullptr
                : &build->memory->cpu_alternate_route_slices[step_index];
        auto alternate = prepare_cpu_job(
            true, build->memory->cpu_alternate_job_slices[step_index],
            alternate_route_slice);
        if (!alternate) {
          prepare.failure = alternate.location();
          prepare.failure.step = declared.logical_step;
          prepare.failure.iteration = declared.iteration;
          return Status::fail(alternate.reason());
        }
        step.alternate_job = std::move(alternate).value();
      }
      continue;
    }
    std::vector<std::shared_ptr<BufferState>> inputs;
    std::vector<std::shared_ptr<BufferState>> outputs;
    std::vector<JobBufferView> input_views;
    std::vector<JobBufferView> output_views;
    inputs.reserve(sealed.inputs.size());
    outputs.reserve(sealed.physical_sources.size());
    input_views.reserve(sealed.inputs.size());
    output_views.reserve(sealed.physical_sources.size());
    for (const PipelineResolvedViewPlan &view : sealed.inputs) {
      inputs.push_back(primary_owner(view.resource));
      input_views.push_back(PipelineScheduleResources::job_view(view));
    }
    for (const std::uint32_t source : sealed.physical_sources) {
      if (source >= sealed.outputs.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const PipelineResolvedViewPlan &view = sealed.outputs[source].view;
      outputs.push_back(primary_owner(view.resource));
      output_views.push_back(PipelineScheduleResources::job_view(view));
    }
    auto job = prepare_pipeline_accel_job(
        step.program, std::move(inputs), std::move(outputs), input_views,
        output_views, build->memory->views[step_index], workspace);
    if (!job) {
      prepare.failure = job.location();
      prepare.failure.step = declared.logical_step;
      prepare.failure.iteration = declared.iteration;
      return Status::fail(job.reason());
    }
    step.job = std::move(job).value();
    if (state->transactional) {
      std::vector<std::shared_ptr<BufferState>> alternate_inputs;
      std::vector<std::shared_ptr<BufferState>> alternate_outputs;
      alternate_inputs.reserve(sealed.inputs.size());
      alternate_outputs.reserve(sealed.physical_sources.size());
      for (const PipelineResolvedViewPlan &view : sealed.inputs) {
        alternate_inputs.push_back(alternate_owner(view.resource));
      }
      for (const std::uint32_t source : sealed.physical_sources) {
        if (source >= sealed.outputs.size()) {
          return Status::fail(Reason::PipelineInvalid);
        }
        alternate_outputs.push_back(
            alternate_owner(sealed.outputs[source].view.resource));
      }
      auto alternate = prepare_pipeline_accel_job(
          step.program, std::move(alternate_inputs),
          std::move(alternate_outputs), std::move(input_views),
          std::move(output_views), build->memory->views[step_index], workspace);
      if (!alternate) {
        prepare.failure = alternate.location();
        prepare.failure.step = declared.logical_step;
        prepare.failure.iteration = declared.iteration;
        return Status::fail(alternate.reason());
      }
      step.alternate_job = std::move(alternate).value();
    }
  }
  const Status publication_bindings = validate_publication_job_bindings(*state);
  if (!publication_bindings) {
    return publication_bindings;
  }
  // The private Jobs now own the exact Buffer owners and Views.  Drop every
  // transient declaration/ordinal table before native stream construction
  // (and before a potentially large restore seed) instead of carrying a
  // second binding authority through the rest of preparation.
  std::vector<PipelineBuildStep>{}.swap(build->steps);
  std::vector<PipelineBuildStatePair>{}.swap(build->state_pairs);
  std::vector<PipelineBuildPublication>{}.swap(build->publications);
  std::vector<PipelineInternal>{}.swap(build->internals);
  std::vector<PipelineBuildWindowControl>{}.swap(build->window_controls);
  std::vector<PipelineBuildNestedWindow>{}.swap(build->nested_windows);
  std::vector<std::shared_ptr<BufferState>>{}.swap(
      build->materialized_resources);
  build->memory.reset();
  state->status_entry_count = status_entry_count;
  state->stats.pipeline.status_entry_count = state->status_entry_count;

  const Status backend = prepare_backend(*state, prepare.failure);
  if (!backend) {
    return Status::fail(backend.reason());
  }
  return Status::success();
}

} // namespace rund::compute::detail
