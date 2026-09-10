#include "../../../backend.hpp"
#include "../scan.hpp"

#include "../../../device/residency/pool.hpp"
#include "../../../pipeline/transfer.hpp"

#include <rund/counter.hpp>

#include <cstring>
#include <span>

namespace rund::compute::detail {
namespace {

using ::rund::detail::counter::Accumulate;

} // namespace

Status upload_virtual_scan_uniform(PipelineState &pipeline,
                                   const VirtualRunProjection &run,
                                   const residency::EpochLease lease,
                                   const VirtualScan &scan, Stats &stats,
                                   const bool restore) noexcept {
  if (pipeline.residency_pool == nullptr || lease.bindings.empty() ||
      lease.bindings.size() > PipelineLeafCapacity ||
      pipeline.residency_bank >= residency::Pool::BankCount) {
    return Status::fail(Reason::PipelineInvalid);
  }
  residency::Pool &pool = *pipeline.residency_pool;
  BufferState *const target = pool.input[pipeline.residency_bank].get();
  const std::size_t element_bytes =
      static_cast<std::size_t>(run.input_page_bytes / run.input_frame_elements);
  if (target == nullptr || element_bytes == 0u ||
      element_bytes > sizeof(std::uint64_t)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::array<UploadRequest, PipelineLeafCapacity> requests{};
  const residency::FrameRegion region =
      pool.input_regions[pipeline.residency_bank];
  const std::uint64_t first = region.first;
  if (region.count != run.frame_capacity) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (std::size_t index = 0u; index < lease.bindings.size(); ++index) {
    const residency::CacheBinding binding = lease.bindings[index];
    if (binding.frame < first || binding.frame >= first + run.frame_capacity) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t local = static_cast<std::size_t>(binding.frame - first);
    requests[index] = UploadRequest{
        .buffer = target,
        .data =
            restore ? scan.original[index].data() : scan.injected[index].data(),
        .bytes = element_bytes,
        .offset = static_cast<std::size_t>(local * run.input_page_bytes),
    };
  }
  UploadResult uploaded{};
  if (pipeline.device->backend == Backend::Cpu) {
    CpuBufferState *const storage = cpu_buffer(*target);
    if (storage == nullptr || storage->data == nullptr) {
      return Status::fail(Reason::TransferInvalid);
    }
    for (std::size_t index = 0u; index < lease.bindings.size(); ++index) {
      std::memcpy(storage->data.get() + requests[index].offset,
                  requests[index].data, element_bytes);
    }
  } else {
    if (pipeline.device->ops == nullptr ||
        pipeline.device->ops->upload_batch == nullptr) {
      return Status::fail(Reason::TransferInvalid);
    }
    uploaded = pipeline.device->ops->upload_batch(
        *pipeline.device,
        std::span<const UploadRequest>{requests.data(), lease.bindings.size()},
        node::accel::detail::TransferCompletion::Complete,
        node::accel::detail::TransferAuthority::PipelinePrivate);
    if (!uploaded.status) {
      return uploaded.status;
    }
  }
  const std::uint64_t bytes = lease.bindings.size() * element_bytes;
  Accumulate(stats.buffer_allocations, uploaded.buffer_allocations);
  Accumulate(stats.buffer_reuses, uploaded.buffer_reuses);
  Accumulate(stats.transfer_submissions.host_to_device,
             uploaded.command_submits);
  if (pipeline.device->backend != Backend::Cpu) {
    Accumulate(stats.uploaded_bytes, bytes);
  }
  Accumulate(stats.host_write_bytes, bytes);
  return Status::success();
}

} // namespace rund::compute::detail
