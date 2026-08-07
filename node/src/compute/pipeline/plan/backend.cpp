#include "local.hpp"
#include "publication.hpp"

#include "../../../accel/kernel/prepared.hpp"
#include "../../../accel/kernel/recurrence.hpp"
#include "../../backend.hpp"
#include "../../buffer/local.hpp"
#include "../../status.hpp"
#include "../local.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool
resolve_view(const PipelineState &state,
             const PipelinePublicationViewPlan &planned, const bool alternate,
             node::accel::detail::BackendRead &resolved) noexcept {
  const PipelinePublicationViewIdentity &identity = planned.identity;
  const PipelineResource *const resource =
      selected_pipeline_resource(state, identity.resource_ordinal, alternate);
  if (state.device == nullptr || state.device->ops == nullptr ||
      state.device->ops->resolve_buffer == nullptr || resource == nullptr ||
      resource->buffer == nullptr || resource->type != planned.type ||
      resource->format != planned.format ||
      resource->bytes != identity.backing_bytes ||
      identity.element_bytes == 0u || identity.stride_bytes == 0u) {
    return false;
  }
  const AccelBufferState *const buffer = accel_buffer(*resource->buffer);
  std::shared_ptr<void> handle;
  if (buffer == nullptr ||
      !state.device->ops
           ->resolve_buffer(*state.device, *resource->buffer, handle)
           .ok()) {
    return false;
  }
  auto ref = buffer->buffer.resident;
  if (ref.bytes != identity.backing_bytes) {
    return false;
  }
  ref.offset_bytes = identity.offset_bytes;
  ref.element_bytes = identity.element_bytes;
  ref.stride_bytes = identity.stride_bytes;
  ref.count = identity.count;
  ref.usage = identity.usage;
  resolved = node::accel::detail::BackendRead{
      .source = ref,
      .handle = std::move(handle),
  };
  return true;
}

[[nodiscard]] bool
supports_window_route(const JobState &job, const PipelineRoute route,
                      const std::uint32_t recurrent_output_count) noexcept {
  if (job.outputs.empty() || job.output_views.size() != job.outputs.size()) {
    return false;
  }
  // Seed and Action own tile intermediates. Only Ordinary recurrence and the
  // nested Fold carry the recurrent prefix; trailing Fold outputs are Window
  // publications and deliberately have no corresponding input coordinate.
  if (route == PipelineRoute::NestedSeed ||
      route == PipelineRoute::NestedAction) {
    return true;
  }
  return recurrent_output_count != 0u &&
         recurrent_output_count <= job.inputs.size() &&
         recurrent_output_count <= job.input_views.size() &&
         recurrent_output_count <= job.outputs.size();
}

} // namespace

Reason
project_pipeline_preparation_reason(const std::string_view reason) noexcept {
  const Reason boundary =
      reason ==
              node::accel::detail::PreparedPipelineTemplateStepCapacityReasonKey
          ? Reason::PipelineCapacity
          : Reason::LoweringInvalid;
  return project_reason(reason, boundary);
}

Status prepare_backend(PipelineState &value, Location &location) noexcept {
  location = {};
  if (value.device == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  if (value.device->backend == Backend::Cpu) {
    return Status::success();
  }
  try {
    PipelineState *const state = &value;
    const DeviceOps *const ops = state->device->ops;
    if (ops == nullptr || ops->prepare_pipeline == nullptr) {
      return Status::fail(Reason::AccelProgramInvalid);
    }
    const auto prepare_stream = [&](const bool alternate)
        -> Result<node::accel::detail::PreparedKernelPipeline> {
      // Pipeline step capacity is a public admission bound.  Keep the cold
      // projection in fixed storage so primary/alternate preparation cannot
      // create an unplanned vector layer before the backend budget gate.
      constexpr std::size_t capacity =
          node::accel::detail::PreparedPipelineStepCapacity;
      std::array<const node::accel::detail::PreparedKernelRun *, capacity>
          prepared{};
      std::array<std::uint8_t, capacity> barriers{};
      std::array<std::uint32_t, capacity> declared_steps{};
      std::array<node::accel::detail::BackendRecurrence, capacity>
          recurrences{};
      std::array<node::accel::detail::BackendWindow, capacity> windows{};
      std::array<node::accel::detail::BackendPublish, PipelineLeafCapacity>
          publications{};
      if (state->steps.size() > capacity ||
          state->publications.size() > publications.size()) {
        return Result<node::accel::detail::PreparedKernelPipeline>::fail(
            Reason::PipelineCapacity);
      }
      if (!state->publications.empty() &&
          (state->device->ops == nullptr ||
           state->device->ops->resolve_buffer == nullptr)) {
        return Result<node::accel::detail::PreparedKernelPipeline>::fail(
            Reason::DeviceInvalid);
      }
      std::size_t active = 0u;
      std::size_t window_count = 0u;
      bool pending_barrier = false;
      for (std::size_t index = 0u; index < state->steps.size(); ++index) {
        pending_barrier = pending_barrier || state->barriers[index] != 0u;
        if (state->steps[index].program->empty()) {
          continue;
        }
        const std::shared_ptr<JobState> &job =
            alternate ? state->steps[index].alternate_job
                      : state->steps[index].job;
        if (job == nullptr) {
          return Result<node::accel::detail::PreparedKernelPipeline>::fail(
              Reason::PipelineInvalid);
        }
        if (active == capacity) {
          return Result<node::accel::detail::PreparedKernelPipeline>::fail(
              Reason::PipelineCapacity);
        }
        prepared[active] = &job->prepared;
        declared_steps[active] = static_cast<std::uint32_t>(index);
        const PipelineStep &step = state->steps[index];
        const node::accel::detail::BackendWindow *window = nullptr;
        if (step.window != 0u) {
          if (step.window > state->windows.size()) {
            return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                Reason::PipelineInvalid);
          }
          const PipelineWindow &declared = state->windows[step.window - 1u];
          const PipelineWindowControl &control = declared.control;
          if (declared.first_step >= state->steps.size() ||
              !supports_window_route(*job, step.route,
                                     declared.recurrent_output_count)) {
            return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                Reason::PipelineInvalid);
          }
          node::accel::detail::BackendRead count;
          if (!resolve_view(*state, control.count, alternate, count)) {
            return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                Reason::BindingDeviceMismatch);
          }
          std::array<node::accel::detail::BackendRead, 3u> terminal{};
          const bool has_terminal = control.terminal_publication !=
                                    std::numeric_limits<std::uint32_t>::max();
          if (has_terminal) {
            if (control.terminal_publication >= state->publications.size()) {
              return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                  Reason::PipelineInvalid);
            }
            const auto *planned = std::get_if<PipelineTerminalPublicationPlan>(
                &state->publications[control.terminal_publication]);
            const std::uint32_t state_index =
                static_cast<std::uint32_t>(step.window - 1u);
            if (planned == nullptr || planned->state != state_index) {
              return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                  Reason::PipelineInvalid);
            }
            for (std::uint32_t bank = PipelineWindow::seed;
                 bank <= PipelineWindow::second; ++bank) {
              if (!resolve_view(*state, planned->sources[bank], alternate,
                                terminal[bank]) ||
                  terminal[bank].source.element_bytes !=
                      sizeof(std::uint32_t) ||
                  terminal[bank].source.count != 1u) {
                return Result<node::accel::detail::PreparedKernelPipeline>::
                    fail(Reason::BindingInvalid);
              }
            }
          }
          using Phase = node::accel::detail::BackendWindowPhase;
          Phase phase = Phase::Ordinary;
          std::uint32_t outer_iteration = step.iteration;
          std::uint32_t outer_bound = step.iteration_bound;
          std::uint32_t inner_iteration = 0u;
          std::uint32_t inner_bound = 1u;
          std::uint32_t inner_advance = 0u;
          std::uint32_t route = 0u;
          if (declared.nested()) {
            node::accel::detail::NestedTemplateRouteProjection projection{};
            if (!declared.nested_shape.project(index, projection) ||
                step.route != pipeline_route(projection.phase) ||
                step.iteration != projection.iteration ||
                step.iteration_bound != projection.bound) {
              return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                  Reason::PipelineInvalid);
            }
            if (!node::accel::detail::ProjectNestedBackendWindowPhase(
                    projection.phase, phase)) {
              return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                  Reason::PipelineInvalid);
            }
            outer_iteration = projection.outer_iteration;
            outer_bound = projection.outer_bound;
            inner_iteration = projection.inner_iteration;
            inner_bound = projection.inner_bound;
            inner_advance = projection.inner_advance;
            route = projection.route;
          } else if (step.route != PipelineRoute::Ordinary) {
            return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                Reason::PipelineInvalid);
          }
          if (window_count == windows.size()) {
            return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                Reason::PipelineCapacity);
          }
          windows[window_count] = node::accel::detail::BackendWindow{
              .count = std::move(count),
              .terminal = std::move(terminal),
              .maximum = control.maximum,
              .tile = control.tile,
              .expected = control.expected,
              .state = static_cast<std::uint32_t>(step.window - 1u),
              .outer_iteration = outer_iteration,
              .outer_bound = outer_bound,
              .inner_iteration = inner_iteration,
              .inner_bound = inner_bound,
              .inner_advance = inner_advance,
              .route = route,
              .phase = phase,
              .has_terminal = has_terminal,
          };
          window = &windows[window_count++];
        }
        recurrences[active] = node::accel::detail::BackendRecurrence{
            .logical_step = state->steps[index].logical_step,
            .iteration = state->steps[index].iteration,
            .bound = state->steps[index].iteration_bound,
            .window = window,
            .writes_each_iteration = state->steps[index].writes_each_iteration,
        };
        barriers[active] =
            active == 0u ? 0u : static_cast<std::uint8_t>(pending_barrier);
        pending_barrier = false;
        ++active;
      }
      for (std::size_t index = 0u; index < state->publications.size();
           ++index) {
        const PipelinePublicationPlan &publication = state->publications[index];
        const auto *window =
            std::get_if<PipelineWindowPublicationPlan>(&publication);
        const auto *terminal =
            std::get_if<PipelineTerminalPublicationPlan>(&publication);
        const std::uint32_t publication_state =
            window != nullptr ? window->state : terminal->state;
        if (publication_state >= state->windows.size()) {
          return Result<node::accel::detail::PreparedKernelPipeline>::fail(
              Reason::PipelineInvalid);
        }
        const PipelineWindow &publication_window =
            state->windows[publication_state];
        const PipelineWindowControl &control = publication_window.control;
        if (control.final < PipelineWindow::first ||
            control.final > PipelineWindow::second) {
          return Result<node::accel::detail::PreparedKernelPipeline>::fail(
              Reason::PipelineInvalid);
        }
        std::array<node::accel::detail::BackendRead, 3u> sources{};
        if (window != nullptr) {
          node::accel::detail::BackendRead source;
          if (!resolve_view(*state, window->source, alternate, source)) {
            return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                Reason::BindingInvalid);
          }
          sources.fill(source);
        } else {
          for (std::uint32_t bank = PipelineWindow::seed;
               bank <= PipelineWindow::second; ++bank) {
            if (!resolve_view(*state, terminal->sources[bank], alternate,
                              sources[bank])) {
              return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                  Reason::BindingInvalid);
            }
          }
        }
        node::accel::detail::BackendRead target;
        node::accel::detail::BackendRead resident_count;
        const PipelinePublicationTargetPlan &target_plan =
            pipeline_publication_target(publication);
        if (!resolve_view(*state, target_plan.view, alternate, target)) {
          return Result<node::accel::detail::PreparedKernelPipeline>::fail(
              Reason::BindingDeviceMismatch);
        }
        if (window != nullptr &&
            !resolve_view(*state, control.count, alternate, resident_count)) {
          return Result<node::accel::detail::PreparedKernelPipeline>::fail(
              Reason::BindingDeviceMismatch);
        }
        if (window != nullptr && (!publication_window.nested() ||
                                  !publication_window.nested_shape.valid())) {
          return Result<node::accel::detail::PreparedKernelPipeline>::fail(
              Reason::PipelineInvalid);
        }
        auto identity = project_pipeline_publication_identity(
            publication, control,
            window == nullptr ? 0u
                              : publication_window.nested_shape.outer_bound());
        for (std::size_t bank = 0u; bank < sources.size(); ++bank) {
          identity.sources[bank].resident_id = sources[bank].source.id;
        }
        identity.target.resident_id = target.source.id;
        if (window != nullptr) {
          identity.count.resident_id = resident_count.source.id;
        }
        publications[index] = node::accel::detail::BackendPublish{
            .sources = std::move(sources),
            .count = std::move(resident_count),
            .target = std::move(target),
            .identity = identity,
        };
      }
      if (active == 0u) {
        state->active_step_count = 0u;
        return Result<node::accel::detail::PreparedKernelPipeline>::success({});
      }
      auto prepared_pipeline = ops->prepare_pipeline(
          *state->device,
          std::span<const node::accel::detail::PreparedKernelRun *const>{
              prepared.data(), active},
          std::span<const std::uint8_t>{barriers.data(), active},
          std::span<const std::uint32_t>{declared_steps.data(), active},
          std::span<const node::accel::detail::BackendRecurrence>{
              recurrences.data(), active},
          std::span<const node::accel::detail::BackendPublish>{
              publications.data(), state->publications.size()},
          static_cast<std::uint32_t>(state->steps.size()),
          state->transactional ? 2u : 1u, state->profile != nullptr,
          &state->accel_templates);
      if (!prepared_pipeline.ok) {
        const node::accel::detail::PreparedPipelineFailure &failure =
            prepared_pipeline.failure;
        location.template_index = failure.template_index;
        location.occurrence_index = failure.occurrence_index;
        location.node = failure.node;
        location.outer_iteration = failure.outer_iteration;
        location.inner_iteration = failure.inner_iteration;
        location.nested_phase = failure.nested_phase;
        location.native_reason_key = failure.native_reason_key;
        if (failure.template_index < active) {
          const std::uint32_t physical = declared_steps[failure.template_index];
          if (physical < state->steps.size()) {
            location.step = state->steps[physical].logical_step;
            location.iteration = state->steps[physical].iteration;
          }
        }
        return Result<node::accel::detail::PreparedKernelPipeline>::fail(
            project_pipeline_preparation_reason(failure.native_reason_key));
      }
      state->active_step_count = static_cast<std::uint32_t>(active);
      return Result<node::accel::detail::PreparedKernelPipeline>::success(
          std::move(prepared_pipeline));
    };
    auto primary = prepare_stream(false);
    if (!primary) {
      return Status::fail(primary.reason());
    }
    state->prepared = std::move(primary).value();
    if (state->transactional) {
      auto alternate = prepare_stream(true);
      if (!alternate) {
        return Status::fail(alternate.reason());
      }
      state->alternate_prepared = std::move(alternate).value();
    }
    return seed_pipeline_generations(*state, 0u, 0u);
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::PipelineCapacity);
  } catch (const std::length_error &) {
    return Status::fail(Reason::PipelineCapacity);
  }
}

} // namespace rund::compute::detail
