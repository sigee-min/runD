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
#include <utility>
#include <vector>

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

} // namespace

Status prepare_backend(PipelineState &value) noexcept {
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
      std::vector<const node::accel::detail::PreparedKernelRun *> prepared;
      std::vector<std::uint8_t> barriers;
      std::vector<std::uint32_t> declared_steps;
      std::vector<node::accel::detail::BackendRecurrence> recurrences;
      std::array<node::accel::detail::BackendPublish, PipelineLeafCapacity>
          publications{};
      prepared.reserve(state->steps.size());
      barriers.reserve(state->steps.size());
      declared_steps.reserve(state->steps.size());
      recurrences.reserve(state->steps.size());
      std::size_t window_count = 0u;
      for (std::size_t index = 0u; index < state->steps.size(); ++index) {
        const PipelineStep &step = state->steps[index];
        if (step.program->empty() || step.window == 0u) {
          continue;
        }
        const std::shared_ptr<JobState> &job =
            alternate ? step.alternate_job : step.job;
        if (job == nullptr) {
          return Result<node::accel::detail::PreparedKernelPipeline>::fail(
              Reason::PipelineInvalid);
        }
        ++window_count;
      }
      std::vector<node::accel::detail::BackendWindow> windows;
      windows.reserve(window_count);
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
        prepared.push_back(&job->prepared);
        declared_steps.push_back(static_cast<std::uint32_t>(index));
        const PipelineStep &step = state->steps[index];
        const node::accel::detail::BackendWindow *window = nullptr;
        if (step.window != 0u) {
          if (step.window > state->windows.size() || job->outputs.empty() ||
              job->outputs.size() > job->inputs.size() ||
              job->input_views.size() < job->outputs.size() ||
              job->output_views.size() != job->outputs.size()) {
            return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                Reason::PipelineInvalid);
          }
          const PipelineWindow &declared = state->windows[step.window - 1u];
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
          windows.push_back(node::accel::detail::BackendWindow{
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
                  phase ==
                          node::accel::detail::BackendWindowPhase::NestedAction
                      ? 1u
                      : 0u,
              .route = route,
              .phase = phase,
              .has_terminal = has_terminal,
          });
          window = &windows.back();
        }
        recurrences.push_back(node::accel::detail::BackendRecurrence{
            .logical_step = state->steps[index].logical_step,
            .iteration = state->steps[index].iteration,
            .bound = state->steps[index].iteration_bound,
            .window = window,
            .writes_each_iteration =
                state->steps[index].writes_each_iteration,
        });
        barriers.push_back(
            active == 0u ? 0u : static_cast<std::uint8_t>(pending_barrier));
        pending_barrier = false;
        ++active;
      }
      for (std::size_t index = 0u; index < state->publications.size();
           ++index) {
        const PipelinePublish &publication = state->publications[index];
        if (publication.window == 0u ||
            publication.window > state->windows.size()) {
          return Result<node::accel::detail::PreparedKernelPipeline>::fail(
              Reason::PipelineInvalid);
        }
        const PipelineWindow &published_window =
            state->windows[publication.window - 1u];
        if (published_window.first_step >= state->steps.size() ||
            state->steps[published_window.first_step].iteration_bound == 0u) {
          return Result<node::accel::detail::PreparedKernelPipeline>::fail(
              Reason::PipelineInvalid);
        }
        std::array<node::accel::detail::BackendRead, 3u> sources{};
        for (std::uint32_t bank = PipelineWindow::seed;
             bank <= PipelineWindow::second; ++bank) {
          if (!resolve_bank(state->windows[publication.window - 1u],
                            publication.output, bank, sources[bank]) ||
              sources[bank].source.count != publication.count ||
              sources[bank].source.element_bytes != publication.element_bytes) {
            return Result<node::accel::detail::PreparedKernelPipeline>::fail(
                Reason::BindingInvalid);
          }
        }
        node::accel::detail::BackendRead target;
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
        if (!resolve_view(*state, *target_owner, publication.target_offset,
                          publication.count, publication.target_stride,
                          publication.element_bytes,
                          kernel::kResidentUsageWrite, target)) {
          return Result<node::accel::detail::PreparedKernelPipeline>::fail(
              Reason::BindingDeviceMismatch);
        }
        publications[index] = node::accel::detail::BackendPublish{
            .sources = std::move(sources),
            .target = target.source,
            .target_handle = std::move(target.handle),
            .state = static_cast<std::uint32_t>(publication.window - 1u),
            .final =
                1u +
                (((published_window.nested
                       ? static_cast<std::uint32_t>(published_window.seed_count)
                       : state->steps[published_window.first_step]
                             .iteration_bound) -
                  1u) &
                 1u),
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
          state->transactional ? 2u : 1u, state->profile != nullptr);
      if (!prepared_pipeline.ok) {
        return Result<node::accel::detail::PreparedKernelPipeline>::fail(
            project_reason(prepared_pipeline.reason, Reason::LoweringInvalid));
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
    return seed_pipeline_generations(*state, 0u);
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::PipelineCapacity);
  }
}

} // namespace rund::compute::detail
