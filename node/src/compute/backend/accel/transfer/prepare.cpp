#include "../local.hpp"

#include "../../../../accel/context/transfer.hpp"
#include "../../../pipeline/state.hpp"
#include "../../../status.hpp"

#include <kernel/core/checked.hpp>

namespace rund::compute::detail::accel_backend {

Status prepare_pipeline_transfer(PipelineState &pipeline) noexcept {
  const AccelDeviceState *const accel =
      pipeline.device == nullptr ? nullptr : accel_device(*pipeline.device);
  const std::uint64_t publication_bytes =
      pipeline.publication == nullptr
          ? 0u
          : pipeline.publication->publication_memory.usage().allocated_bytes;
  std::uint64_t admitted_bytes = 0u;
  if (accel == nullptr || !pipeline.private_memory.committed() ||
      !kernel::checked::add(pipeline.private_memory.usage().allocated_bytes,
                            publication_bytes, admitted_bytes) ||
      admitted_bytes != pipeline.plan.committed_peak_bytes ||
      pipeline.residency_transfer_committed_bytes == 0u ||
      pipeline.residency_input >= pipeline.resources.size() ||
      pipeline.residency_output >= pipeline.resources.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const PipelineResource &input = pipeline.resources[pipeline.residency_input];
  const PipelineResource &output =
      pipeline.resources[pipeline.residency_output];
  const AccelBufferState *const input_storage =
      input.buffer == nullptr ? nullptr : accel_buffer(*input.buffer);
  const AccelBufferState *const output_storage =
      output.buffer == nullptr ? nullptr : accel_buffer(*output.buffer);
  if (input_storage == nullptr || output_storage == nullptr ||
      input.bytes == 0u || output.bytes == 0u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const rund::AccelCheck prepared =
      node::accel::detail::PrepareAccelPipelineTransfer(
          accel->context, pipeline.prepared, input_storage->buffer,
          output_storage->buffer, input.bytes, output.bytes,
          pipeline.residency_transfer_committed_bytes);
  return prepared.ok ? Status::success()
                     : Status::fail(project_reason(prepared.reason,
                                                   Reason::TransferInvalid));
}

} // namespace rund::compute::detail::accel_backend
