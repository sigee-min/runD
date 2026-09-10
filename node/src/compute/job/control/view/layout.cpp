#include "../../../cpu/state/program.hpp"
#include "../../../device/state.hpp"
#include "../model.hpp"

#include "local.hpp"

#include "../../../cpu/graph.hpp"
#include "../../../cpu/run/state.hpp"
#include "../../../exception.hpp"
#include "../../../type.hpp"

#include <kernel/core/checked.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace rund::compute::detail {

bool cpu_view_transfer_bytes(const JobBufferView view, const Type type,
                             std::uint64_t &bytes) noexcept {
  if (view.element_bytes != type_bytes(type)) {
    return false;
  }
  return kernel::checked::mul(static_cast<std::uint64_t>(view.count),
                              static_cast<std::uint64_t>(view.element_bytes),
                              bytes);
}

namespace {

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
        if (!cpu_view_transfer_bytes(view, types[index], bytes) ||
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

} // namespace rund::compute::detail
