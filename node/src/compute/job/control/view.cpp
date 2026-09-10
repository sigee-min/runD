#include "../../cpu/state/program.hpp"
#include "../../device/state.hpp"
#include "model.hpp"

#include "view/local.hpp"

#include "../../cpu/graph.hpp"
#include "../../cpu/run/state.hpp"
#include "../../exception.hpp"

#include <cstdint>
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

} // namespace rund::compute::detail
