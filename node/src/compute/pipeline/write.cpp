#include <rund/compute/pipeline.hpp>

#include <rund/counter.hpp>

#include "../buffer/local.hpp"
#include "state.hpp"
#include "transfer.hpp"

#include <mutex>

namespace rund::compute::detail {

Status write_pipeline_raw(const std::shared_ptr<PipelineState> &state,
                          const std::shared_ptr<BufferState> &buffer,
                          const HostView input, WriteStats &writes) noexcept {
  if (!valid_pipeline(state) || buffer == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{state->gate};
  if (state->phase != PipelinePhase::Ready) {
    return Status::fail(state->phase == PipelinePhase::Poisoned
                            ? Reason::PipelinePoisoned
                            : Reason::PipelineBusy);
  }

  UploadResult transfer{};
  const Status written = write_buffer_measured(buffer, input, writes, transfer);
  if (input.count == 0u) {
    return written;
  }

  if (state->samples != PipelineState::SampleState::Inactive) {
    state->samples = PipelineState::SampleState::Dirty;
  }
  if (buffer->device->backend == Backend::Cpu) {
    if (written) {
      record_pipeline_transfer(*state, buffer->bytes);
      record_transfer(*state->device, buffer->bytes);
    }
  } else {
    const bool physical_attempt = written || transfer.staging_bytes != 0u ||
                                  transfer.staging_peak_bytes != 0u ||
                                  transfer.staging_reused_bytes != 0u ||
                                  transfer.buffer_allocations != 0u ||
                                  transfer.buffer_reuses != 0u ||
                                  transfer.command_submits != 0u;
    if (physical_attempt) {
      record_pipeline_upload(*state, buffer->bytes, transfer);
    }
  }
  if (written) {
    ::rund::detail::counter::Accumulate(state->stats.host_write_bytes,
                                        buffer->bytes);
  }
  return written;
}

} // namespace rund::compute::detail
