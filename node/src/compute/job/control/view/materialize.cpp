#include "../../../cpu/state/program.hpp"
#include "../../../../../include/rund/compute/abi/device.hpp"
#include "../model.hpp"

#include "local.hpp"

#include "../../../buffer/local.hpp"
#include "../../../cpu/run/state.hpp"
#include "../../../exception.hpp"
#include "../../../type.hpp"

#include <kernel/core/checked.hpp>

#include <memory>
#include <utility>

namespace rund::compute::detail {

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
                           auto &transfers, const bool input) -> Status {
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
          !cpu_view_transfer_bytes(view, external->type, bytes) ||
          bytes != slot.bytes ||
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

} // namespace rund::compute::detail
