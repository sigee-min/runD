#include "model.hpp"

#include <rund/counter.hpp>
#include "../../buffer/local.hpp"
#include "../../cpu/graph.hpp"
#include "../../cpu/run/state.hpp"
#include "../../cpu/view.hpp"
#include "../../type.hpp"
#include "../local.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool whole_view(const BufferState &buffer,
                              const JobBufferView view) noexcept {
  return view.offset == 0u && view.count == buffer.count && view.stride == 1u &&
         view.element_bytes == type_bytes(buffer.type);
}

void mark_dense_value(const ProgramState &program, const std::uint32_t value,
                      std::vector<bool> &inputs,
                      std::vector<bool> &outputs) noexcept {
  if (value == 0u || value - 1u >= program.graph_value_routes.size()) {
    return;
  }
  const GraphValueRoute route = program.graph_value_routes[value - 1u];
  if (route.source == GraphBindSource::Input && route.index < inputs.size()) {
    inputs[route.index] = true;
  } else if (route.source == GraphBindSource::Output &&
             route.index < outputs.size()) {
    outputs[route.index] = true;
  }
}

} // namespace

Status prepare_cpu_view_transfers(JobState &state) {
  if (state.program->device->backend != Backend::Cpu ||
      state.output_views.empty() || state.program->cpu_graph == nullptr ||
      state.program->cpu_graph->runtime == nullptr) {
    return Status::success();
  }
  std::vector<bool> dense_inputs(state.inputs.size());
  std::vector<bool> dense_outputs(state.outputs.size());
  const CpuRuntimeGraph &runtime = *state.program->cpu_graph->runtime;
  for (std::size_t step_index = 0u; step_index < runtime.steps.size();
       ++step_index) {
    const CpuRuntimeStep &step = runtime.steps[step_index];
    if (const auto *const map = std::get_if<CpuRuntimeMap>(&step)) {
      mark_dense_value(*state.program, map->control.predicate, dense_inputs,
                       dense_outputs);
      mark_dense_value(*state.program, map->control.count, dense_inputs,
                       dense_outputs);
      continue;
    }
    if (const auto *const scan = std::get_if<CpuRuntimeScan>(&step)) {
      mark_dense_value(*state.program, scan->input, dense_inputs,
                       dense_outputs);
      mark_dense_value(*state.program, scan->output, dense_inputs,
                       dense_outputs);
      mark_dense_value(*state.program, scan->count, dense_inputs,
                       dense_outputs);
      continue;
    }
    if (step_index >= state.program->cpu_graph->bind_begin.size() ||
        step_index >= state.program->cpu_graph->bind_count.size()) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    const std::size_t begin = state.program->cpu_graph->bind_begin[step_index];
    const std::size_t count = state.program->cpu_graph->bind_count[step_index];
    if (begin > state.program->graph_bindings.size() ||
        count > state.program->graph_bindings.size() - begin) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    for (std::size_t index = 0u; index < count; ++index) {
      const GraphRunBinding binding =
          state.program->graph_bindings[begin + index];
      mark_dense_value(*state.program, binding.value_index + 1u, dense_inputs,
                       dense_outputs);
    }
  }

  const auto prepare =
      [&](std::vector<std::shared_ptr<BufferState>> &owners,
          std::vector<JobBufferView> &views, const std::vector<bool> &dense,
          std::vector<CpuViewTransfer> &transfers, const bool input) -> Status {
    for (std::size_t index = 0u; index < owners.size(); ++index) {
      const std::shared_ptr<BufferState> external = owners[index];
      const JobBufferView view = views[index];
      const bool needs_staging = view.count > 1u && view.stride != 1u;
      if (!dense[index] || external == nullptr || whole_view(*external, view) ||
          !needs_staging) {
        continue;
      }
      Result<std::shared_ptr<BufferState>> made =
          input
              ? make_input_binding_buffer(state.program->device, external->type,
                                          view.count)
              : make_buffer(state.program->device, external->type, view.count);
      if (!made) {
        return Status::fail(made.reason());
      }
      std::shared_ptr<BufferState> staging = std::move(made).value();
      transfers.push_back(CpuViewTransfer{
          .external = external, .staging = staging, .view = view});
      state.cpu_view_buffers.push_back(staging);
      owners[index] = std::move(staging);
      const std::size_t element_bytes = type_bytes(external->type);
      views[index] = JobBufferView{.count = view.count,
                                   .stride = 1u,
                                   .element_bytes = element_bytes,
                                   .alignment = element_bytes};
    }
    return Status::success();
  };
  state.cpu_view_inputs.reserve(state.inputs.size());
  state.cpu_view_outputs.reserve(state.outputs.size());
  state.cpu_view_buffers.reserve(state.inputs.size() + state.outputs.size());
  const Status inputs = prepare(state.inputs, state.input_views, dense_inputs,
                                state.cpu_view_inputs, true);
  return inputs ? prepare(state.outputs, state.output_views, dense_outputs,
                          state.cpu_view_outputs, false)
                : inputs;
}

namespace {

Status copy_cpu_view(const CpuViewTransfer &transfer, const bool publish,
                     std::size_t &bytes) noexcept {
  bytes = 0u;
  const std::optional<CpuView> external =
      cpu_view(transfer.external.get(), transfer.view);
  const std::optional<CpuView> staging =
      cpu_view(transfer.staging.get(), 0u, transfer.view.count, 1u,
               transfer.view.element_bytes);
  if (!external || !staging ||
      external->footprint.bytes != staging->footprint.bytes) {
    return Status::fail(Reason::CpuBufferInvalid);
  }
  bytes = external->footprint.bytes;
  if (bytes == 0u) {
    return Status::success();
  }
  if (external->data == nullptr || staging->data == nullptr) {
    return Status::fail(Reason::CpuBufferInvalid);
  }
  if (external->footprint.dense()) {
    if (publish) {
      std::memcpy(external->data, staging->data, bytes);
    } else {
      std::memcpy(staging->data, external->data, bytes);
    }
    return Status::success();
  }

  std::byte *external_data = external->data;
  std::byte *staging_data = staging->data;
  for (std::size_t remaining = external->footprint.count; remaining > 1u;
       --remaining) {
    if (publish) {
      std::memcpy(external_data, staging_data, external->footprint.width);
    } else {
      std::memcpy(staging_data, external_data, external->footprint.width);
    }
    external_data += external->footprint.stride;
    staging_data += staging->footprint.stride;
  }
  if (publish) {
    std::memcpy(external_data, staging_data, external->footprint.width);
  } else {
    std::memcpy(staging_data, external_data, external->footprint.width);
  }
  return Status::success();
}

} // namespace

Status
gather_cpu_pipeline_views(const std::shared_ptr<JobState> &state) noexcept {
  if (state == nullptr || state->program == nullptr ||
      state->program->device == nullptr ||
      state->program->device->backend != Backend::Cpu) {
    return Status::fail(Reason::PipelineInvalid);
  }
  state->cpu_view_gather_bytes = 0u;
  for (const CpuViewTransfer &transfer : state->cpu_view_inputs) {
    std::size_t bytes = 0u;
    const Status copied = copy_cpu_view(transfer, false, bytes);
    if (!copied) {
      return copied;
    }
    state->cpu_view_gather_bytes = ::rund::detail::counter::SaturatingAdd(
        state->cpu_view_gather_bytes, bytes);
  }
  return Status::success();
}

Status
publish_cpu_pipeline_views(const std::shared_ptr<JobState> &state) noexcept {
  if (state == nullptr || state->program == nullptr ||
      state->program->device == nullptr ||
      state->program->device->backend != Backend::Cpu) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (state->cpu == nullptr) {
    return state->cpu_view_inputs.empty() && state->cpu_view_outputs.empty() &&
                   state->cpu_view_gather_bytes == 0u
               ? Status::success()
               : Status::fail(Reason::CpuRunInvalid);
  }
  std::uint64_t bytes = state->cpu_view_gather_bytes;
  state->cpu_view_gather_bytes = 0u;
  for (const CpuViewTransfer &transfer : state->cpu_view_outputs) {
    std::size_t copied_bytes = 0u;
    const Status copied = copy_cpu_view(transfer, true, copied_bytes);
    if (!copied) {
      return copied;
    }
    bytes = ::rund::detail::counter::SaturatingAdd(bytes, copied_bytes);
  }
  ::rund::detail::counter::Accumulate(
      state->cpu->stats.internal_roundtrip_bytes, bytes);
  return Status::success();
}

} // namespace rund::compute::detail
