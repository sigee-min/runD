#include "../internal.hpp"

#include <array>
#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck
FinalizeMetalProjectionWindows(MetalPipelineBuild &build,
                               MetalPipelineFinalizeProjection &projection) {
  const std::uint32_t no_declared_step =
      std::numeric_limits<std::uint32_t>::max();
  bool direct_window =
      !build.native_windows.empty() && !build.recurrence.ready() &&
      build.transducers.empty() && !build.aggregate_selected &&
      !build.profile_steps &&
      build.native_windows.size() == build.entries.size() &&
      build.native_windows.size() == build.pipeline->residency_steps.size() &&
      build.entries.size() == build.status.declared_step_count;
  if (direct_window) {
    for (std::size_t index = 0u; index < build.native_windows.size(); ++index) {
      const BackendBatchEntry &entry = build.entries[index];
      const BackendWindow *const window = entry.recurrence.window;
      const auto *const resources =
          static_cast<const MetalKernelResources *>(entry.prepared->get());
      const std::uint32_t declared =
          entry.template_index < build.status.active_step_count
              ? build.status.declared_steps[entry.template_index]
              : no_declared_step;
      direct_window =
          direct_window && window != nullptr && resources != nullptr &&
          entry.transducer == NoTileTransducer && !resources->shared_scratch &&
          window->phase == BackendWindowPhase::Ordinary &&
          window->route == 0u && window->outer_iteration == 0u &&
          window->outer_bound == 1u && window->inner_iteration == 0u &&
          window->inner_bound == 1u && window->inner_advance == 0u &&
          window->valid_occurrence(false) && declared != no_declared_step &&
          declared < build.pipeline->residency_steps.size() &&
          build.native_windows[index].entry == index &&
          build.native_windows[index].params.declared_step == declared;
      for (std::size_t prior = 0u; direct_window && prior < index; ++prior) {
        direct_window =
            build.native_windows[prior].params.declared_step != declared &&
            build.native_windows[prior].params.state !=
                build.native_windows[index].params.state;
      }
    }
    for (const MetalPublish &publication : build.native_publication_rows()) {
      direct_window =
          direct_window && publication.params.kind ==
                               static_cast<std::uint32_t>(
                                   PreparedKernelPublicationKind::Terminal);
    }
    direct_window = direct_window &&
                    build.pipeline->state_count == build.native_windows.size();
  }
  bool spatial_window =
      build.pipeline->spatial_window.admitted && !build.recurrence.ready() &&
      build.recurrence.history == nullptr && build.transducers.empty() &&
      !build.aggregate_selected && !build.profile_steps &&
      build.native_publication_count == 0u &&
      build.entries.size() == build.status.declared_step_count &&
      build.pipeline->spatial_window.local_count ==
          build.status.declared_step_count &&
      build.pipeline->spatial_window.shape_valid();
  if (spatial_window) {
    std::array<bool, PreparedPipelineStepCapacity> local_seen{};
    for (const BackendBatchEntry &entry : build.entries) {
      spatial_window = spatial_window && entry.run != nullptr &&
                       build.pipeline->spatial_window.matches(*entry.run);
      if (!spatial_window ||
          entry.template_index >= build.status.active_step_count) {
        break;
      }
      const std::uint32_t declared =
          build.status.declared_steps[entry.template_index];
      if (declared >= build.pipeline->spatial_window.local_count ||
          local_seen[declared]) {
        spatial_window = false;
        break;
      }
      local_seen[declared] = true;
      const BoundStep &producer_step = entry.run->steps[0u];
      const BoundStep &window_step = entry.run->steps[1u];
      const BoundStep &consumer_step = entry.run->steps[2u];
      const StepBinds *const producer_binds =
          BindingsFor<StepBinds>(producer_step, rund::kernel::NodeKind::Map);
      const RangeBinds *const bindings =
          BindingsFor<RangeBinds>(window_step, rund::kernel::NodeKind::Window);
      const StepBinds *const consumer_binds =
          BindingsFor<StepBinds>(consumer_step, rund::kernel::NodeKind::Map);
      if (producer_binds == nullptr || consumer_binds == nullptr ||
          !producer_binds->valid() || !consumer_binds->valid() ||
          producer_binds->inputs.size() != 1u ||
          producer_binds->outputs.size() != 1u ||
          consumer_binds->inputs.size() != 1u ||
          consumer_binds->outputs.size() != 1u || bindings == nullptr ||
          bindings->input == nullptr || bindings->input_handle == nullptr ||
          *bindings->input_handle == nullptr || bindings->output == nullptr ||
          bindings->output_handle == nullptr ||
          *bindings->output_handle == nullptr ||
          window_step.source_binds == nullptr ||
          window_step.source_binds->refs() == nullptr ||
          window_step.source_binds->handles() == nullptr ||
          build.pipeline->spatial_window.window_input_binding >=
              window_step.source_binds->size() ||
          build.pipeline->spatial_window.window_output_binding >=
              window_step.source_binds->size() ||
          &window_step.source_binds->refs()[build.pipeline->spatial_window
                                                .window_input_binding] !=
              bindings->input ||
          &window_step.source_binds->refs()[build.pipeline->spatial_window
                                                .window_output_binding] !=
              bindings->output ||
          &window_step.source_binds->handles()[build.pipeline->spatial_window
                                                   .window_input_binding] !=
              bindings->input_handle ||
          &window_step.source_binds->handles()[build.pipeline->spatial_window
                                                   .window_output_binding] !=
              bindings->output_handle) {
        spatial_window = false;
        break;
      }
      // ResidentBindingRange is a small view; retain the concrete ranges so
      // the row can authenticate the exact producer/consumer bindings after
      // preparation rather than trusting graph indices alone.
      const rund::kernel::ResidentBindingRange producer_inputs =
          producer_binds->inputs.range();
      const rund::kernel::ResidentBindingRange producer_outputs =
          producer_binds->outputs.range();
      const rund::kernel::ResidentBindingRange consumer_inputs =
          consumer_binds->inputs.range();
      const rund::kernel::ResidentBindingRange consumer_outputs =
          consumer_binds->outputs.range();
      const std::array<const rund::kernel::ResidentBindingRange *, 4u>
          concrete_views{&producer_inputs, &producer_outputs, &consumer_inputs,
                         &consumer_outputs};
      MetalSpatialWindowLocalProof &local =
          build.pipeline->spatial_window.locals[declared];
      local.execution = entry.run->execution;
      local.input = *bindings->input;
      local.output = *bindings->output;
      local.input_handle = bindings->input_handle->get();
      local.output_handle = bindings->output_handle->get();
      local.bound_step = &window_step;
      local.producer_step = &producer_step;
      local.consumer_step = &consumer_step;
      local.binding_owner = window_step.source_binds;
      local.input_index = build.pipeline->spatial_window.binding_indices[2u];
      local.output_index = build.pipeline->spatial_window.binding_indices[3u];
      local.input_binding_index =
          build.pipeline->spatial_window.window_input_binding;
      local.output_binding_index =
          build.pipeline->spatial_window.window_output_binding;
      local.edge_steps = std::array<const void *, 3u>{
          &producer_step, &window_step, &consumer_step};
      bool edge_bindings = true;
      const std::array<std::size_t, 4u> edge_positions{0u, 1u, 4u, 5u};
      for (std::size_t edge = 0u; edge < concrete_views.size(); ++edge) {
        const rund::kernel::ResidentBufferRef *const ref =
            concrete_views[edge]->ref(0u);
        const std::shared_ptr<void> *const handle =
            concrete_views[edge]->handle(0u);
        edge_bindings = edge_bindings && ref != nullptr && handle != nullptr &&
                        *handle != nullptr;
        if (edge_bindings) {
          local.edge_refs[edge_positions[edge]] = *ref;
          local.edge_handles[edge_positions[edge]] = handle->get();
        }
      }
      local.edge_refs[2u] = *bindings->input;
      local.edge_refs[3u] = *bindings->output;
      local.edge_handles[2u] = local.input_handle;
      local.edge_handles[3u] = local.output_handle;
      local.edge_bindings_captured = edge_bindings;
      local.captured = true;
      if (!local.valid()) {
        spatial_window = false;
        break;
      }
    }
    if (spatial_window) {
      for (std::size_t local = 0u;
           local < build.pipeline->spatial_window.local_count; ++local) {
        spatial_window = spatial_window && local_seen[local];
      }
    }
  }
  build.pipeline->spatial_window.local_bindings_complete = spatial_window;
  projection.direct_window = direct_window;
  projection.spatial_window = spatial_window;
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
