#include "local.hpp"

#include "../../../accel/kernel/prepared.hpp"
#include "../../backend.hpp"
#include "../../buffer/local.hpp"
#include "../../size.hpp"
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

[[nodiscard]] bool resolve_view(
    const PipelineState &state, const std::shared_ptr<BufferState> &owner,
    const std::size_t offset, const std::size_t count, const std::size_t stride,
    const std::size_t element_bytes, const std::uint32_t usage,
    node::accel::detail::BackendRead &resolved) noexcept {
  std::size_t offset_bytes = 0u;
  std::size_t stride_bytes = 0u;
  if (state.device == nullptr || state.device->ops == nullptr ||
      state.device->ops->resolve_buffer == nullptr || owner == nullptr ||
      element_bytes == 0u || stride == 0u ||
      !size::multiply(offset, element_bytes, offset_bytes) ||
      !size::multiply(stride, element_bytes, stride_bytes)) {
    return false;
  }
  const AccelBufferState *const buffer = accel_buffer(*owner);
  std::shared_ptr<void> handle;
  if (buffer == nullptr ||
      !state.device->ops->resolve_buffer(*state.device, *owner, handle).ok()) {
    return false;
  }
  auto ref = buffer->buffer.resident;
  ref.offset_bytes = offset_bytes;
  ref.element_bytes = element_bytes;
  ref.stride_bytes = stride_bytes;
  ref.count = count;
  ref.usage = usage;
  resolved = node::accel::detail::BackendRead{
      .source = ref,
      .handle = std::move(handle),
  };
  return true;
}

[[nodiscard]] bool
resolve_view(const PipelineState &state,
             const std::shared_ptr<BufferState> &owner,
             const JobBufferView view, const std::uint32_t usage,
             node::accel::detail::BackendRead &resolved) noexcept {
  return resolve_view(state, owner, view.offset, view.count, view.stride,
                      view.element_bytes, usage, resolved);
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

Reason project_pipeline_preparation_reason(
    const std::string_view reason) noexcept {
  const Reason boundary =
      reason == node::accel::detail::
                    PreparedPipelineTemplateStepCapacityReasonKey
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
      const auto resolve_bank =
          [&](const PipelineWindow &window, const std::uint32_t output,
              const std::uint32_t bank,
              node::accel::detail::BackendRead &resolved) {
            if (bank > PipelineWindow::second ||
                window.first_step >= state->steps.size()) {
              return false;
            }
            std::size_t step_index = window.first_step;
            bool input = bank == PipelineWindow::seed;
            if (bank == PipelineWindow::second) {
              if (step_index + 1u < state->steps.size() &&
                  state->steps[step_index + 1u].window ==
                      state->steps[step_index].window) {
                ++step_index;
              }
            }
            const PipelineStep &step = state->steps[step_index];
            const std::shared_ptr<JobState> &job =
                alternate ? step.alternate_job : step.job;
            if (job == nullptr) {
              return false;
            }
            if (input) {
              return output < job->inputs.size() &&
                     output < job->input_views.size() &&
                     resolve_view(*state, job->inputs[output],
                                  job->input_views[output],
                                  kernel::kResidentUsageRead, resolved);
            }
            return output < job->outputs.size() &&
                   output < job->output_views.size() &&
                   resolve_view(*state, job->outputs[output],
                                job->output_views[output],
                                kernel::kResidentUsageRead, resolved);
      };
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
          if (declared.first_step >= state->steps.size() ||
              !supports_window_route(*job, step.route,
                                     declared.recurrent_output_count)) {
            return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                Reason::PipelineInvalid);
          }
          node::accel::detail::BackendRead count;
          if (!resolve_view(*state, declared.count, declared.count_offset, 1u,
                            1u, sizeof(std::uint32_t),
                            kernel::kResidentUsageRead, count)) {
            return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                Reason::BindingDeviceMismatch);
          }
          std::array<node::accel::detail::BackendRead, 3u> terminal{};
          const bool has_terminal =
              declared.terminal != std::numeric_limits<std::uint32_t>::max();
          if (has_terminal) {
            for (std::uint32_t bank = PipelineWindow::seed;
                 bank <= PipelineWindow::second; ++bank) {
              if (!resolve_bank(declared, declared.terminal_output, bank,
                                terminal[bank]) ||
                  terminal[bank].source.element_bytes !=
                      sizeof(std::uint32_t) ||
                  terminal[bank].source.count != 1u) {
                return Result<node::accel::detail::PreparedKernelPipeline>::
                    fail(Reason::BindingInvalid);
              }
            }
          }
          const auto phase = [&] {
            using Phase = node::accel::detail::BackendWindowPhase;
            switch (step.route) {
            case PipelineRoute::NestedSeed:
              return Phase::NestedSeed;
            case PipelineRoute::NestedAction:
              return Phase::NestedAction;
            case PipelineRoute::NestedFold:
              return Phase::NestedFold;
            case PipelineRoute::Ordinary:
              return Phase::Ordinary;
            }
            return Phase::Ordinary;
          }();
          const std::uint32_t outer_iteration =
              step.route == PipelineRoute::NestedSeed ||
                      step.route == PipelineRoute::Ordinary
                  ? step.iteration
                  : 0u;
          const std::uint32_t outer_bound =
              declared.nested ? static_cast<std::uint32_t>(declared.seed_count)
                              : step.iteration_bound;
          const std::uint32_t inner_iteration =
              step.route == PipelineRoute::NestedAction ? step.iteration : 0u;
          const std::uint32_t inner_bound =
              declared.nested
                  ? static_cast<std::uint32_t>(declared.action_count)
                  : 1u;
          const std::uint32_t route =
              step.route == PipelineRoute::NestedFold ? step.iteration : 0u;
          if (window_count == windows.size()) {
            return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                Reason::PipelineCapacity);
          }
          windows[window_count] = node::accel::detail::BackendWindow{
              .count = std::move(count),
              .terminal = std::move(terminal),
              .maximum = declared.maximum,
              .tile = declared.tile,
              .iteration = outer_iteration,
              .bound = outer_bound,
              .expected = declared.expected,
              .state = static_cast<std::uint32_t>(step.window - 1u),
              .outer_iteration = outer_iteration,
              .outer_bound = outer_bound,
              .inner_iteration = inner_iteration,
              .inner_bound = inner_bound,
              .inner_advance =
                  phase == node::accel::detail::BackendWindowPhase::NestedAction
                      ? 1u
                      : 0u,
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
        const PipelinePublish &publication = state->publications[index];
        if (publication.window == 0u ||
            publication.window > state->windows.size() ||
            publication.state != publication.window - 1u) {
          return Result<node::accel::detail::PreparedKernelPipeline>::fail(
              Reason::PipelineInvalid);
        }
        const PipelineWindow &published_window =
            state->windows[publication.window - 1u];
        const bool window_publish =
            publication.kind == PipelinePublishKind::Window;
        if (published_window.first_step >= state->steps.size() ||
            state->steps[published_window.first_step].iteration_bound == 0u ||
            (window_publish ? publication.final != 0u
                            : publication.final < PipelineWindow::first ||
                                  publication.final > PipelineWindow::second)) {
          return Result<node::accel::detail::PreparedKernelPipeline>::fail(
              Reason::PipelineInvalid);
        }
        std::array<node::accel::detail::BackendRead, 3u> sources{};
        if (window_publish) {
          node::accel::detail::BackendRead source;
          if (!resolve_view(*state, publication.source,
                            publication.source_offset, publication.count, 1u,
                            publication.element_bytes,
                            kernel::kResidentUsageRead, source) ||
              source.source.count != publication.count ||
              source.source.element_bytes != publication.element_bytes) {
            return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                Reason::BindingInvalid);
          }
          sources.fill(source);
        } else {
          for (std::uint32_t bank = PipelineWindow::seed;
               bank <= PipelineWindow::second; ++bank) {
            if (!resolve_bank(state->windows[publication.window - 1u],
                              publication.output, bank, sources[bank]) ||
                sources[bank].source.count != publication.count ||
                sources[bank].source.element_bytes !=
                    publication.element_bytes) {
              return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                  Reason::BindingInvalid);
            }
          }
        }
        node::accel::detail::BackendRead target;
        node::accel::detail::BackendRead resident_count;
        const std::shared_ptr<BufferState> *target_owner = &publication.target;
        if (alternate) {
          const auto canonical =
              std::find_if(state->resources.begin(), state->resources.end(),
                           [&](const PipelineResource &resource) {
                             return resource.buffer == publication.target;
                           });
          const std::size_t ordinal =
              static_cast<std::size_t>(canonical - state->resources.begin());
          if (canonical == state->resources.end() ||
              state->alternate_claims.size() != state->resources.size() ||
              state->alternate_claims[ordinal].buffer == nullptr) {
            return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                Reason::PipelineInvalid);
          }
          const BufferState *const alternate_target =
              state->alternate_claims[ordinal].buffer;
          const auto owner =
              std::find_if(state->resources.begin(), state->resources.end(),
                           [&](const PipelineResource &resource) {
                             return resource.buffer.get() == alternate_target;
                           });
          if (owner == state->resources.end()) {
            return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                Reason::PipelineInvalid);
          }
          target_owner = &owner->buffer;
        }
        const std::size_t target_count =
            window_publish ? publication.maximum : publication.count;
        if (!resolve_view(*state, *target_owner, publication.target_offset,
                          target_count, publication.target_stride,
                          publication.element_bytes,
                          kernel::kResidentUsageWrite, target)) {
          return Result<node::accel::detail::PreparedKernelPipeline>::fail(
              Reason::BindingDeviceMismatch);
        }
        if (window_publish &&
            !resolve_view(*state, publication.resident_count,
                          publication.resident_count_offset, 1u, 1u,
                          sizeof(std::uint32_t), kernel::kResidentUsageRead,
                          resident_count)) {
          return Result<node::accel::detail::PreparedKernelPipeline>::fail(
              Reason::BindingDeviceMismatch);
        }
        publications[index] = node::accel::detail::BackendPublish{
            .sources = std::move(sources),
            .source_ordinals = publication.ordinals.sources,
            .count = std::move(resident_count),
            .count_ordinal = publication.ordinals.count,
            .target = target.source,
            .target_handle = std::move(target.handle),
            .target_ordinal = publication.ordinals.target,
            .state = publication.state,
            .final = publication.final,
            .maximum = publication.maximum,
            .tile = publication.tile,
            .kind = window_publish
                        ? node::accel::detail::BackendPublishKind::Window
                        : node::accel::detail::BackendPublishKind::Terminal,
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
