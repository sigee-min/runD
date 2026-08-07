#include "model.hpp"

#include "../../buffer/local.hpp"
#include "../../cpu/graph.hpp"
#include "../../cpu/run/state.hpp"
#include "../../cpu/view.hpp"
#include "../../exception.hpp"
#include "../../type.hpp"
#include "../local.hpp"
#include <rund/counter.hpp>

#include <kernel/core/checked.hpp>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool
mark_dense_value(const ProgramState &program, const std::uint32_t value,
                 std::vector<std::uint8_t> &inputs,
                 std::vector<std::uint8_t> &outputs) noexcept {
  if (value == 0u) {
    return true;
  }
  if (value - 1u >= program.graph_value_routes.size()) {
    return false;
  }
  const GraphValueRoute route = program.graph_value_routes[value - 1u];
  if (route.source == GraphBindSource::Input) {
    if (route.index >= inputs.size()) {
      return false;
    }
    inputs[route.index] = true;
  } else if (route.source == GraphBindSource::Output) {
    if (route.index >= outputs.size()) {
      return false;
    }
    outputs[route.index] = true;
  } else if (route.source != GraphBindSource::Internal) {
    return false;
  }
  return true;
}

[[nodiscard]] bool transfer_bytes(const JobBufferView view, const Type type,
                                  std::uint64_t &bytes) noexcept {
  if (view.element_bytes != type_bytes(type)) {
    return false;
  }
  return kernel::checked::mul(static_cast<std::uint64_t>(view.count),
                              static_cast<std::uint64_t>(view.element_bytes),
                              bytes);
}

[[nodiscard]] bool
valid_requirement_indices(const std::vector<std::uint32_t> &indices,
                          const std::size_t count) noexcept {
  std::uint32_t previous = 0u;
  bool first = true;
  for (const std::uint32_t index : indices) {
    if (index >= count || (!first && index <= previous)) {
      return false;
    }
    first = false;
    previous = index;
  }
  return true;
}

} // namespace

Result<CpuViewTransferRequirements> plan_cpu_view_transfer_requirements(
    const std::shared_ptr<ProgramState> &program) noexcept {
  if (program == nullptr || program->device == nullptr) {
    return Result<CpuViewTransferRequirements>::fail(Reason::ProgramInvalid);
  }
  CpuViewTransferRequirements requirements{
      .program = program.get(),
      .input_count = program->input_types.size(),
      .output_count = program->output_types.size(),
      .inputs = {},
      .outputs = {},
  };
  if (program->device->backend != Backend::Cpu ||
      program->output_types.empty() || program->cpu_graph == nullptr ||
      program->cpu_graph->runtime == nullptr) {
    return Result<CpuViewTransferRequirements>::success(
        std::move(requirements));
  }
  requirements.graph_hash = program->cpu_graph->graph_hash;
  try {
    std::vector<std::uint8_t> dense_inputs(program->input_types.size());
    std::vector<std::uint8_t> dense_outputs(program->output_types.size());
    const CpuRuntimeGraph &runtime = *program->cpu_graph->runtime;
    for (std::size_t step_index = 0u; step_index < runtime.steps.size();
         ++step_index) {
      const CpuRuntimeStep &step = runtime.steps[step_index];
      if (const auto *const map = std::get_if<CpuRuntimeMap>(&step)) {
        if (!mark_dense_value(*program, map->control.predicate, dense_inputs,
                              dense_outputs) ||
            !mark_dense_value(*program, map->control.count, dense_inputs,
                              dense_outputs)) {
          return Result<CpuViewTransferRequirements>::fail(
              Reason::GraphBindingInvalid);
        }
        continue;
      }
      if (const auto *const scan = std::get_if<CpuRuntimeScan>(&step)) {
        if (!mark_dense_value(*program, scan->input, dense_inputs,
                              dense_outputs) ||
            !mark_dense_value(*program, scan->output, dense_inputs,
                              dense_outputs) ||
            !mark_dense_value(*program, scan->count, dense_inputs,
                              dense_outputs)) {
          return Result<CpuViewTransferRequirements>::fail(
              Reason::GraphBindingInvalid);
        }
        continue;
      }
      if (step_index >= program->cpu_graph->bind_begin.size() ||
          step_index >= program->cpu_graph->bind_count.size()) {
        return Result<CpuViewTransferRequirements>::fail(
            Reason::GraphBindingInvalid);
      }
      const std::size_t begin = program->cpu_graph->bind_begin[step_index];
      const std::size_t count = program->cpu_graph->bind_count[step_index];
      if (begin > program->graph_bindings.size() ||
          count > program->graph_bindings.size() - begin) {
        return Result<CpuViewTransferRequirements>::fail(
            Reason::GraphBindingInvalid);
      }
      for (std::size_t index = 0u; index < count; ++index) {
        const GraphRunBinding binding = program->graph_bindings[begin + index];
        if (binding.value_index >= program->graph_value_routes.size() ||
            !mark_dense_value(*program, binding.value_index + 1u, dense_inputs,
                              dense_outputs)) {
          return Result<CpuViewTransferRequirements>::fail(
              Reason::GraphBindingInvalid);
        }
      }
    }
    const auto append = [](const std::vector<std::uint8_t> &dense,
                           std::vector<std::uint32_t> &indices) {
      indices.reserve(dense.size());
      for (std::size_t index = 0u; index < dense.size(); ++index) {
        if (dense[index] != 0u) {
          if (index > std::numeric_limits<std::uint32_t>::max()) {
            return false;
          }
          indices.push_back(static_cast<std::uint32_t>(index));
        }
      }
      return true;
    };
    if (!append(dense_inputs, requirements.inputs) ||
        !append(dense_outputs, requirements.outputs)) {
      return Result<CpuViewTransferRequirements>::fail(
          Reason::PipelineCapacity);
    }
    return Result<CpuViewTransferRequirements>::success(
        std::move(requirements));
  } catch (...) {
    compute_exception::rethrow_unless_capacity_exception();
    return Result<CpuViewTransferRequirements>::fail(Reason::PipelineCapacity);
  }
}

Result<CpuViewTransferLayout> plan_cpu_view_transfers(
    const std::shared_ptr<ProgramState> &program,
    const std::span<const JobBufferView> input_views,
    const std::span<const JobBufferView> output_views,
    const CpuViewTransferRequirements *requirements) noexcept {
  if (program == nullptr || program->device == nullptr) {
    return Result<CpuViewTransferLayout>::fail(Reason::ProgramInvalid);
  }
  CpuViewTransferLayout plan{.program = program.get(),
                             .input_count = input_views.size(),
                             .output_count = output_views.size(),
                             .inputs = {},
                             .outputs = {}};
  if (program->device->backend != Backend::Cpu || output_views.empty() ||
      program->cpu_graph == nullptr || program->cpu_graph->runtime == nullptr) {
    return Result<CpuViewTransferLayout>::success(std::move(plan));
  }
  if (input_views.size() != program->input_types.size() ||
      output_views.size() != program->output_types.size()) {
    return Result<CpuViewTransferLayout>::fail(Reason::BindingCountMismatch);
  }
  CpuViewTransferRequirements planned;
  if (requirements == nullptr) {
    auto result = plan_cpu_view_transfer_requirements(program);
    if (!result) {
      return Result<CpuViewTransferLayout>::fail(result.reason());
    }
    planned = std::move(result).value();
    requirements = &planned;
  }
  plan.graph_hash = program->cpu_graph->graph_hash;
  if (requirements->program != program.get() ||
      requirements->graph_hash != plan.graph_hash ||
      requirements->input_count != input_views.size() ||
      requirements->output_count != output_views.size() ||
      !valid_requirement_indices(requirements->inputs, input_views.size()) ||
      !valid_requirement_indices(requirements->outputs, output_views.size())) {
    return Result<CpuViewTransferLayout>::fail(Reason::PipelineInvalid);
  }
  try {
    const auto append = [&](const std::span<const JobBufferView> views,
                            const std::span<const Type> types,
                            const std::vector<std::uint32_t> &indices,
                            std::vector<CpuViewTransferSlot> &slots) {
      slots.reserve(indices.size());
      for (const std::uint32_t index : indices) {
        const JobBufferView view = views[index];
        if (view.count <= 1u || view.stride == 1u) {
          continue;
        }
        std::uint64_t bytes = 0u;
        if (!transfer_bytes(view, types[index], bytes) ||
            !kernel::checked::add(plan.bytes, bytes, plan.bytes)) {
          return false;
        }
        slots.push_back(CpuViewTransferSlot{.index = index, .bytes = bytes});
      }
      return true;
    };
    if (!append(input_views, program->input_types, requirements->inputs,
                plan.inputs) ||
        !append(output_views, program->output_types, requirements->outputs,
                plan.outputs)) {
      return Result<CpuViewTransferLayout>::fail(Reason::PipelineCapacity);
    }
    return Result<CpuViewTransferLayout>::success(std::move(plan));
  } catch (...) {
    compute_exception::rethrow_unless_capacity_exception();
    return Result<CpuViewTransferLayout>::fail(Reason::PipelineCapacity);
  }
}

Status prepare_cpu_view_transfers(JobState &state,
                                  const CpuViewTransferLayout *layout) {
  if (state.input_views.empty() && state.output_views.empty() &&
      layout == nullptr) {
    return Status::success();
  }
  CpuViewTransferLayout planned;
  if (layout == nullptr) {
    auto result = plan_cpu_view_transfers(state.program, state.input_views,
                                          state.output_views);
    if (!result) {
      return Status::fail(result.reason());
    }
    planned = std::move(result).value();
    layout = &planned;
  }
  const CpuGraphProgram *const graph =
      state.program == nullptr ? nullptr : state.program->cpu_graph.get();
  if (state.program == nullptr || state.program->device == nullptr ||
      layout->program != state.program.get() ||
      layout->input_count != state.inputs.size() ||
      layout->input_count != state.input_views.size() ||
      layout->output_count != state.outputs.size() ||
      layout->output_count != state.output_views.size() ||
      (graph != nullptr && layout->graph_hash != graph->graph_hash) ||
      (graph == nullptr && layout->graph_hash != 0u)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (layout->inputs.empty() && layout->outputs.empty()) {
    return layout->bytes == 0u ? Status::success()
                               : Status::fail(Reason::PipelineInvalid);
  }
  if (state.program->device->backend != Backend::Cpu || graph == nullptr ||
      graph->runtime == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::uint64_t expected_bytes = 0u;
  const auto prepare = [&](auto &owners, auto &views,
                           const std::vector<CpuViewTransferSlot> &slots,
                           auto &transfers,
                           const bool input) -> Status {
    std::uint32_t previous = 0u;
    bool first = true;
    for (const CpuViewTransferSlot slot : slots) {
      const std::size_t index = slot.index;
      if (index >= owners.size() || (!first && slot.index <= previous)) {
        return Status::fail(Reason::PipelineInvalid);
      }
      first = false;
      previous = slot.index;
      const std::shared_ptr<BufferState> external = owners[index];
      const JobBufferView view = views[index];
      std::uint64_t bytes = 0u;
      if (external == nullptr || view.count <= 1u || view.stride == 1u ||
          !transfer_bytes(view, external->type, bytes) || bytes != slot.bytes ||
          !kernel::checked::add(expected_bytes, bytes, expected_bytes)) {
        return Status::fail(Reason::PipelineInvalid);
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
          .external = external, .view = view, .binding = slot.index});
      owners[index] = std::move(staging);
      const std::size_t element_bytes = type_bytes(external->type);
      views[index] = JobBufferView{.count = view.count,
                                   .stride = 1u,
                                   .element_bytes = element_bytes,
                                   .alignment = element_bytes};
    }
    return Status::success();
  };
  try {
    state.cpu_view_inputs.reserve(layout->inputs.size());
    state.cpu_view_outputs.reserve(layout->outputs.size());
    const Status inputs = prepare(state.inputs, state.input_views,
                                  layout->inputs, state.cpu_view_inputs, true);
    if (!inputs) {
      return inputs;
    }
    const Status outputs =
        prepare(state.outputs, state.output_views, layout->outputs,
                state.cpu_view_outputs, false);
    return outputs && expected_bytes == layout->bytes
               ? Status::success()
               : (outputs ? Status::fail(Reason::PipelineInvalid) : outputs);
  } catch (...) {
    compute_exception::rethrow_unless_capacity_exception();
    return Status::fail(Reason::BufferCapacity);
  }
}

namespace {

Status copy_cpu_view(
    const CpuViewTransfer &transfer,
    const std::span<const std::shared_ptr<BufferState>> staging_owners,
    const bool publish, std::size_t &bytes) noexcept {
  bytes = 0u;
  if (transfer.binding >= staging_owners.size()) {
    return Status::fail(Reason::CpuBufferInvalid);
  }
  const std::optional<CpuView> external =
      cpu_view(transfer.external.get(), transfer.view);
  const std::optional<CpuView> staging =
      cpu_view(staging_owners[transfer.binding].get(), 0u,
               transfer.view.count, 1u,
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
    const Status copied = copy_cpu_view(transfer, state->inputs, false, bytes);
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
    const Status copied =
        copy_cpu_view(transfer, state->outputs, true, copied_bytes);
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
