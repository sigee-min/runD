#include "local.hpp"

#include "../arena.hpp"
#include "../compare.hpp"
#include "../prepare.hpp"
#include "../resource.hpp"

#include "../../../../accel/kernel/recurrence.hpp"
#include "../../../backend.hpp"
#include "../../../buffer/local.hpp"
#include "../../../cpu/run/state.hpp"
#include "../../../job/local.hpp"
#include "../../../memory/arena.hpp"
#include "../../../status.hpp"
#include "../../../type.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/abi/observe.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rund::compute::detail {

namespace {
[[nodiscard]] Status append_program_binding_identity(
    const PipelineResolvedViewPlan &view, const std::uint32_t usage,
    std::vector<node::accel::detail::PreparedKernelProgramBindingIdentity>
        &out) {
  if (view.element_bytes == 0u || view.count == 0u || view.stride_bytes == 0u) {
    return Status::fail(Reason::PipelineCapacity);
  }
  out.push_back(node::accel::detail::PreparedKernelProgramBindingIdentity{
      .offset_bytes = view.offset_bytes,
      .element_bytes = view.element_bytes,
      .stride_bytes = view.stride_bytes,
      .count = view.count,
      .usage = usage,
  });
  return Status::success();
}

[[nodiscard]] Status plan_program_binding_identities(
    const PipelineBuildStep &step, const std::size_t step_index,
    const PipelineMemoryPlan &plan,
    std::vector<node::accel::detail::PreparedKernelProgramBindingIdentity>
        &out) {
  const ProgramState *const program = step.program.get();
  if (program == nullptr || step_index >= plan.workspace_routes.size() ||
      step_index >= plan.step_resources.size() ||
      plan.steps.size() != plan.workspace_routes.size() + 1u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const PipelineStepResourcePlan &sealed = plan.step_resources[step_index];
  out.clear();
  out.reserve(program->graph_bindings.size());
  for (const GraphRunBinding graph_binding : program->graph_bindings) {
    if (graph_binding.value_index >= program->graph_value_routes.size()) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    const std::uint32_t usage =
        graph_binding.role == kernel::BufferRole::Read
            ? kernel::kResidentUsageRead
        : graph_binding.role == kernel::BufferRole::Write
            ? kernel::kResidentUsageWrite
            : 0u;
    if (usage == 0u) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    const GraphValueRoute route =
        program->graph_value_routes[graph_binding.value_index];
    const PipelineResolvedViewPlan *view = nullptr;
    if (route.source == GraphBindSource::Input) {
      if (route.index >= sealed.inputs.size()) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
      view = &sealed.inputs[route.index];
    } else if (route.source == GraphBindSource::Output) {
      if (route.index >= sealed.physical_sources.size()) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
      const std::uint32_t source = sealed.physical_sources[route.index];
      if (source >= sealed.outputs.size()) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
      view = &sealed.outputs[source].view;
    }
    if (view != nullptr) {
      const Status appended =
          append_program_binding_identity(*view, usage, out);
      if (!appended) {
        return appended;
      }
      continue;
    }
    if (route.source != GraphBindSource::Internal ||
        route.index >= program->chunks.size() || route.element_bytes == 0u ||
        route.count == 0u) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    const PipelineWorkspaceRoute workspace = plan.workspace_routes[step_index];
    if (!workspace.present()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t workspace_index = workspace.owner;
    if (workspace_index >= step_index + 1u ||
        workspace_index + 1u >= plan.steps.size() ||
        workspace_index >= plan.workspace_routes.size() ||
        !plan.workspace_routes[workspace_index].owns(workspace_index)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t begin = plan.steps[workspace_index];
    const std::size_t end = plan.steps[workspace_index + 1u];
    if (begin > end || end > plan.offsets.size() ||
        end - begin != program->chunk_order.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    std::size_t rank = 0u;
    while (rank < program->chunk_order.size() &&
           program->chunk_order[rank] != route.index) {
      ++rank;
    }
    std::uint64_t workspace_offset = 0u;
    std::uint64_t offset_bytes = 0u;
    if (rank == program->chunk_order.size() ||
        !kernel::checked::mul(
            static_cast<std::uint64_t>(plan.offsets[begin + rank]),
            static_cast<std::uint64_t>(memory::Word), workspace_offset) ||
        !kernel::checked::add(workspace_offset, route.offset_bytes,
                              offset_bytes)) {
      return Status::fail(Reason::PipelineCapacity);
    }
    out.push_back(node::accel::detail::PreparedKernelProgramBindingIdentity{
        .offset_bytes = offset_bytes,
        .element_bytes = route.element_bytes,
        .stride_bytes = route.element_bytes,
        .count = route.count,
        .usage = usage,
    });
  }
  return out.size() == program->graph_bindings.size()
             ? Status::success()
             : Status::fail(Reason::GraphBindingInvalid);
}
} // namespace

[[nodiscard]] Status
plan_pipeline_accel_preparation(const PipelineBuildState &build,
                                PipelineMemoryPlan &plan) {
  plan.accel_preparation = {};
  if (build.device->backend == Backend::Cpu) {
    return Status::success();
  }
  const DeviceOps *const ops = build.device->ops;
  if (ops == nullptr || ops->plan_pipeline_preparation == nullptr ||
      plan.job_owners.size() != build.steps.size() ||
      plan.workspace_routes.size() != build.steps.size() ||
      plan.views.size() != build.steps.size() ||
      plan.window_states.size() != build.steps.size()) {
    return Status::fail(Reason::DeviceInvalid);
  }

  try {
    std::vector<std::uint64_t> entry_counts(build.steps.size(), 0u);
    std::vector<std::uint64_t> occurrence_counts(build.steps.size(), 0u);
    std::vector<std::uint64_t> window_counts(build.steps.size(), 0u);
    std::vector<std::uint64_t> nested_group_counts(build.steps.size(), 0u);
    std::vector<std::uint64_t> map_recurrence_group_counts(build.steps.size(),
                                                           0u);
    std::vector<std::uint64_t> map_recurrence_history_group_counts(
        build.steps.size(), 0u);
    std::vector<std::uint64_t> recurrence_hi(build.steps.size(), 0u);
    std::vector<std::uint64_t> recurrence_lo(build.steps.size(), 0u);
    std::vector<std::uint8_t> active_window_states(build.steps.size(), 0u);
    std::uint64_t window_state_count = 0u;
    std::uint64_t window_descriptor_state_count = 0u;
    for (std::size_t index = 0u; index < build.steps.size(); ++index) {
      node::accel::detail::SeedPreparedKernelRecurrenceFingerprint(
          recurrence_hi[index], recurrence_lo[index]);
    }
    for (std::size_t index = 0u; index < build.steps.size(); ++index) {
      const PipelineBuildStep &step = build.steps[index];
      if (step.program == nullptr || step.program->empty()) {
        continue;
      }
      const std::size_t owner = plan.job_owners[index];
      if (owner >= build.steps.size() || owner > index) {
        return Status::fail(Reason::PipelineInvalid);
      }
      node::accel::detail::PreparedKernelRecurrenceIdentity identity{
          .logical_step = step.logical_step,
          .iteration = step.iteration,
          .bound = step.iteration_bound,
          .writes_each_iteration = step.writes_each_iteration,
      };
      constexpr std::uint32_t unassigned =
          std::numeric_limits<std::uint32_t>::max();
      const std::uint32_t state = plan.window_states[index];
      const PipelineWindowControl *control = nullptr;
      if (state != unassigned) {
        if (state >= plan.window_controls.size()) {
          return Status::fail(Reason::PipelineInvalid);
        }
        control = &plan.window_controls[state];
      }
      std::uint64_t occurrences = 1u;
      if (step.nested != 0u) {
        const std::size_t nested_index = step.nested - 1u;
        if (nested_index >= build.nested_windows.size()) {
          return Status::fail(Reason::PipelineInvalid);
        }
        const PipelineBuildNestedWindow &nested =
            build.nested_windows[nested_index];
        node::accel::detail::NestedTemplateRouteProjection route{};
        if (control == nullptr || !nested.shape.valid() ||
            !nested.shape.project(index, route) ||
            !node::accel::detail::ProjectNestedRecurrenceIdentity(
                nested.shape, index,
                node::accel::detail::NestedTemplateRecurrenceIdentityBase{
                    .logical_step = step.logical_step,
                    .maximum = control->maximum,
                    .tile = control->tile,
                    .expected = control->expected,
                    .state = state,
                    .has_terminal = control->terminal !=
                                    std::numeric_limits<std::uint32_t>::max(),
                },
                identity) ||
            step.route != pipeline_route(route.phase) ||
            step.iteration != identity.iteration ||
            step.iteration_bound != identity.bound ||
            step.writes_each_iteration) {
          return Status::fail(Reason::PipelineInvalid);
        }
        occurrences = route.occurrence_count;
      } else if (control != nullptr) {
        identity.maximum = control->maximum;
        identity.tile = control->tile;
        identity.expected = control->expected;
        identity.outer_iteration = step.iteration;
        identity.outer_bound = step.iteration_bound;
        identity.inner_bound = 1u;
        identity.phase = node::accel::detail::BackendWindowPhase::Ordinary;
        identity.has_window = true;
        identity.has_terminal =
            control->terminal != std::numeric_limits<std::uint32_t>::max();
      }
      if (identity.has_window) {
        if (state == unassigned || state >= active_window_states.size()) {
          return Status::fail(Reason::PipelineInvalid);
        }
        identity.state = state;
        window_state_count = std::max(window_state_count,
                                      static_cast<std::uint64_t>(state) + 1u);
        if (active_window_states[state] == 0u) {
          active_window_states[state] = 1u;
          ++window_descriptor_state_count;
        }
      }
      if (!node::accel::detail::MixPreparedKernelRecurrenceFingerprint(
              recurrence_hi[owner], recurrence_lo[owner], identity)) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (!kernel::checked::add(entry_counts[owner], 1u, entry_counts[owner]) ||
          !kernel::checked::add(occurrence_counts[owner], occurrences,
                                occurrence_counts[owner]) ||
          (identity.has_window &&
           !kernel::checked::add(window_counts[owner], occurrences,
                                 window_counts[owner]))) {
        return Status::fail(Reason::PipelineCapacity);
      }
    }
    for (const PipelineBuildNestedWindow &nested : build.nested_windows) {
      const node::accel::detail::NestedTemplateShape &shape = nested.shape;
      if (!shape.valid() || shape.seed_first() >= build.steps.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const std::size_t owner = plan.job_owners[shape.seed_first()];
      if (owner >= build.steps.size() ||
          build.steps[shape.seed_first()].program == nullptr ||
          build.steps[shape.seed_first()].program->empty()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (!kernel::checked::add(nested_group_counts[owner], 1u,
                                nested_group_counts[owner])) {
        return Status::fail(Reason::PipelineCapacity);
      }
      if (shape.action_group_candidate()) {
        if (shape.action_first() >= build.steps.size()) {
          return Status::fail(Reason::PipelineInvalid);
        }
        const std::size_t action_owner = plan.job_owners[shape.action_first()];
        if (action_owner >= build.steps.size() ||
            !kernel::checked::add(map_recurrence_group_counts[action_owner], 1u,
                                  map_recurrence_group_counts[action_owner])) {
          return Status::fail(action_owner >= build.steps.size()
                                  ? Reason::PipelineInvalid
                                  : Reason::PipelineCapacity);
        }
      }
    }

    // A top-level Map recurrence is a whole-command-stream transform. It is
    // therefore one candidate group at most, never one group per iteration or
    // parity route. Runtime proves bindings/artifact eligibility against this
    // same authored marker before materializing any native recurrence owner.
    bool top_level_recurrence =
        build.steps.size() > 1u &&
        build.steps.size() <= std::numeric_limits<std::uint32_t>::max();
    const PipelineBuildStep *const top =
        top_level_recurrence ? &build.steps.front() : nullptr;
    for (std::size_t index = 0u;
         top_level_recurrence && index < build.steps.size(); ++index) {
      const PipelineBuildStep &step = build.steps[index];
      top_level_recurrence =
          step.program == top->program &&
          step.logical_step == top->logical_step && step.iteration == index &&
          step.iteration_bound == build.steps.size() && step.nested == 0u &&
          step.route == PipelineRoute::Ordinary &&
          plan.window_states[index] == PipelineResourceUnassigned &&
          step.writes_each_iteration == top->writes_each_iteration;
    }
    if (top_level_recurrence) {
      const std::size_t owner = plan.job_owners.front();
      if (owner >= build.steps.size() ||
          !kernel::checked::add(map_recurrence_group_counts[owner], 1u,
                                map_recurrence_group_counts[owner]) ||
          (top->writes_each_iteration &&
           !kernel::checked::add(map_recurrence_history_group_counts[owner], 1u,
                                 map_recurrence_history_group_counts[owner]))) {
        return Status::fail(owner >= build.steps.size()
                                ? Reason::PipelineInvalid
                                : Reason::PipelineCapacity);
      }
    }

    std::vector<node::accel::detail::PreparedKernelProgramRoute> routes;
    std::vector<
        std::vector<node::accel::detail::PreparedKernelProgramBindingIdentity>>
        route_program_bindings;
    routes.reserve(build.steps.size());
    route_program_bindings.reserve(build.steps.size());
    const std::uint32_t route_copies = build.state_pairs.empty() ? 1u : 2u;
    const bool has_arena = !plan.view_chunks.empty() || !plan.scratch.empty();
    for (std::size_t index = 0u; index < build.steps.size(); ++index) {
      const PipelineBuildStep &step = build.steps[index];
      if (plan.job_owners[index] != index || step.program == nullptr ||
          step.program->empty()) {
        continue;
      }
      const PipelineWorkspaceRoute workspace = plan.workspace_routes[index];
      if ((has_arena && !workspace.present()) ||
          (workspace.present() &&
           (workspace.owner > index || workspace.owner >= build.steps.size() ||
            !plan.workspace_routes[workspace.owner].owns(workspace.owner) ||
            build.steps[workspace.owner].program != step.program))) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (step.program->accel == nullptr || entry_counts[index] == 0u) {
        return Status::fail(Reason::AccelProgramInvalid);
      }
      route_program_bindings.emplace_back();
      const Status binding_identity = plan_program_binding_identities(
          step, index, plan, route_program_bindings.back());
      if (!binding_identity) {
        return binding_identity;
      }
      routes.push_back(node::accel::detail::PreparedKernelProgramRoute{
          .kernel = &step.program->accel->kernel,
          .tile_count = step.program->count,
          .views = has_arena ? &plan.views[index] : nullptr,
          .scratch = has_arena ? &plan.scratch : nullptr,
          .program_bindings = route_program_bindings.back(),
          .entry_count = entry_counts[index],
          .occurrence_count = occurrence_counts[index],
          .window_count = window_counts[index],
          .nested_group_count = nested_group_counts[index],
          .map_recurrence_group_count = map_recurrence_group_counts[index],
          .map_recurrence_history_group_count =
              map_recurrence_history_group_counts[index],
          .recurrence_fingerprint_hi = recurrence_hi[index],
          .recurrence_fingerprint_lo = recurrence_lo[index],
          .route_copies = route_copies,
      });
    }
    if (routes.empty()) {
      // A Pipeline may still contain declared logical steps when every
      // Program is canonical zero work.  No accelerator route, template, or
      // native command exists in that case, but bind() still consumes the
      // frozen planning decision.  Preserve that distinction as an admitted
      // zero reservation instead of leaving the default (invalid) sentinel.
      plan.accel_preparation.ok = true;
      plan.accel_preparation.reason = "ok";
      return Status::success();
    }

    node::accel::detail::PreparedKernelTemplateRegistry templates{};
    std::uint64_t terminal_publication_count = 0u;
    std::uint64_t publication_command_count = 0u;
    for (const PipelinePublicationPlan &publication : plan.publications) {
      const auto *window =
          std::get_if<PipelineWindowPublicationPlan>(&publication);
      const std::uint32_t state = std::visit(
          [](const auto &typed) { return typed.state; }, publication);
      if (state >= plan.window_controls.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const node::accel::detail::NestedTemplateShape *const nested_shape =
          window != nullptr ? pipeline_build_nested_shape(build, state)
                            : nullptr;
      if (window != nullptr &&
          (nested_shape == nullptr || !nested_shape->valid())) {
        return Status::fail(Reason::PipelineInvalid);
      }
      std::uint64_t contribution = 0u;
      if (!node::accel::detail::PreparedKernelPublicationCommandContribution(
              window == nullptr
                  ? node::accel::detail::PreparedKernelPublicationKind::Terminal
                  : node::accel::detail::PreparedKernelPublicationKind::Window,
              nested_shape == nullptr ? 0u : nested_shape->outer_bound(),
              contribution) ||
          !kernel::checked::add(publication_command_count, contribution,
                                publication_command_count) ||
          (window == nullptr &&
           !kernel::checked::add(terminal_publication_count, 1u,
                                 terminal_publication_count))) {
        return Status::fail(Reason::PipelineCapacity);
      }
    }
    plan.accel_preparation = ops->plan_pipeline_preparation(
        *build.device, routes,
        node::accel::detail::PreparedKernelPipelineShape{
            .publication_count = plan.publications.size(),
            .terminal_publication_count = terminal_publication_count,
            .backend_publication_command_count = publication_command_count,
            .window_state_count = window_state_count,
            .window_descriptor_state_count = window_descriptor_state_count,
            .publication_fingerprint_hi = plan.publication_fingerprint_hi,
            .publication_fingerprint_lo = plan.publication_fingerprint_lo,
            .declared_step_count = build.steps.size(),
            .route_copies = route_copies,
            .profile_steps = build.profile == PipelineProfile::Steps,
        },
        templates);
    if (!plan.accel_preparation.ok || !templates.limit.ok ||
        plan.accel_preparation.fingerprint_hi !=
            templates.limit.fingerprint_hi ||
        plan.accel_preparation.fingerprint_lo !=
            templates.limit.fingerprint_lo) {
      return Status::fail(project_reason(plan.accel_preparation.reason,
                                         Reason::LoweringInvalid));
    }
    if (!kernel::checked::add(plan.summary.prepared_host_bytes,
                              plan.accel_preparation.host_bytes,
                              plan.summary.prepared_host_bytes) ||
        !kernel::checked::add(plan.summary.prepared_native_bytes,
                              plan.accel_preparation.native_bytes,
                              plan.summary.prepared_native_bytes)) {
      return Status::fail(Reason::PipelineCapacity);
    }
    return Status::success();
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::PipelineCapacity);
  }
}
} // namespace rund::compute::detail
