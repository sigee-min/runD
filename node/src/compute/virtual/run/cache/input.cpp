#include "../../../backend.hpp"
#include "internal.hpp"

namespace rund::compute::detail {

BufferWriteView residency_input_view(PipelineState &pipeline,
                                     const VirtualRunProjection &run) noexcept {
  if (pipeline.device == nullptr || pipeline.device->backend == Backend::Cpu ||
      pipeline.device->ops == nullptr ||
      pipeline.device->ops->host_write == nullptr ||
      pipeline.residency_pool == nullptr ||
      pipeline.residency_stage != PipelineResidencyStage::Direct || run.scan ||
      run.reduction || run.graph_reduction ||
      pipeline.residency_bank >= residency::Pool::BankCount ||
      pipeline.residency_input >= pipeline.resources.size()) {
    return {};
  }
  PipelineResource &resource = pipeline.resources[pipeline.residency_input];
  const std::shared_ptr<BufferState> &owner =
      pipeline.residency_pool->input[pipeline.residency_bank];
  if (!resource.owned || resource.buffer == nullptr ||
      resource.buffer != owner || resource.buffer->device != pipeline.device ||
      resource.bytes != run.input_arena_bytes ||
      resource.buffer->bytes != run.input_arena_bytes) {
    return {};
  }
  const BufferWriteView view =
      pipeline.device->ops->host_write(*pipeline.device, *resource.buffer);
  return !view || view.bytes != resource.buffer->bytes ? BufferWriteView{}
                                                       : view;
}

} // namespace rund::compute::detail
