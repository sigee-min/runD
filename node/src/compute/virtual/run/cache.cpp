#include "cache.hpp"

#include "../../device/residency_pool.hpp"
#include "../../pipeline/run/clock.hpp"
#include "../../pipeline/transfer.hpp"
#include "../backing.hpp"

#include <rund/counter.hpp>

#include <array>
#include <cstring>
#include <limits>
#include <span>

namespace rund::compute::detail {
namespace {

using ::rund::detail::counter::Accumulate;

[[nodiscard]] Status
copy_frames(PipelineState &pipeline, const BufferState &source,
            BufferState &target,
            const std::span<const residency::CacheBinding> bindings,
            const std::uint64_t source_page_bytes,
            const std::uint64_t target_page_bytes, Stats &stats) noexcept {
  if (bindings.empty() || bindings.size() > PipelineLeafCapacity ||
      source_page_bytes == 0u || source_page_bytes != target_page_bytes) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::array<CopyRequest, PipelineLeafCapacity> requests{};
  for (std::size_t index = 0u; index < bindings.size(); ++index) {
    const residency::CacheBinding binding = bindings[index];
    if (binding.frame >= pipeline.residency_pool->layout.frame_capacity ||
        binding.frame >
            std::numeric_limits<std::size_t>::max() / source_page_bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t offset =
        static_cast<std::size_t>(binding.frame * source_page_bytes);
    requests[index] =
        CopyRequest{.source = &source,
                    .target = &target,
                    .bytes = static_cast<std::size_t>(source_page_bytes),
                    .source_offset = offset,
                    .target_offset = offset};
  }
  CopyResult copied{};
  if (pipeline.device->backend == Backend::Cpu) {
    const CpuBufferState *const source_cpu = cpu_buffer(source);
    CpuBufferState *const target_cpu = cpu_buffer(target);
    if (source_cpu == nullptr || target_cpu == nullptr ||
        source_cpu->data == nullptr || target_cpu->data == nullptr) {
      return Status::fail(Reason::TransferInvalid);
    }
    for (std::size_t index = 0u; index < bindings.size(); ++index) {
      const CopyRequest request = requests[index];
      if (request.source_offset > source_cpu->bytes ||
          request.bytes > source_cpu->bytes - request.source_offset ||
          request.target_offset > target_cpu->bytes ||
          request.bytes > target_cpu->bytes - request.target_offset) {
        return Status::fail(Reason::TransferInvalid);
      }
      std::memcpy(target_cpu->data.get() + request.target_offset,
                  source_cpu->data.get() + request.source_offset,
                  request.bytes);
    }
  } else {
    if (pipeline.device->ops == nullptr ||
        pipeline.device->ops->copy_batch == nullptr) {
      return Status::fail(Reason::TransferInvalid);
    }
    copied = pipeline.device->ops->copy_batch(
        *pipeline.device,
        std::span<const CopyRequest>{requests.data(), bindings.size()},
        node::accel::detail::TransferAuthority::PipelinePrivate);
    if (!copied.status) {
      return copied.status;
    }
  }
  Accumulate(stats.transfer_submissions.device_to_device,
             copied.command_submits);
  return Status::success();
}

[[nodiscard]] UploadResult
upload_cache(PipelineState &pipeline, BufferState &buffer,
             const VirtualRunProjection &run,
             const std::span<const residency::CacheBinding> bindings) noexcept {
  std::array<UploadRequest, PipelineLeafCapacity> requests{};
  std::size_t count = 0u;
  for (const residency::CacheBinding binding : bindings) {
    if (!binding.fetch) {
      continue;
    }
    const std::size_t offset =
        static_cast<std::size_t>(binding.frame * run.input_page_bytes);
    requests[count++] = UploadRequest{
        .buffer = &buffer,
        .data = run.input_stage + offset,
        .bytes = static_cast<std::size_t>(run.input_page_bytes),
        .offset = offset,
    };
  }
  if (count == 0u) {
    return {};
  }
  if (pipeline.device->backend == Backend::Cpu) {
    CpuBufferState *const storage = cpu_buffer(buffer);
    if (storage == nullptr || storage->data == nullptr) {
      return UploadResult{.status = Status::fail(Reason::TransferInvalid)};
    }
    for (std::size_t index = 0u; index < count; ++index) {
      const UploadRequest request = requests[index];
      if (request.offset > storage->bytes ||
          request.bytes > storage->bytes - request.offset) {
        return UploadResult{.status = Status::fail(Reason::TransferInvalid)};
      }
      std::memcpy(storage->data.get() + request.offset, request.data,
                  request.bytes);
    }
    return {};
  }
  if (pipeline.device->ops == nullptr ||
      pipeline.device->ops->upload_batch == nullptr) {
    return UploadResult{.status = Status::fail(Reason::TransferInvalid)};
  }
  return pipeline.device->ops->upload_batch(
      *pipeline.device, std::span<const UploadRequest>{requests.data(), count},
      node::accel::detail::TransferCompletion::Complete,
      node::accel::detail::TransferAuthority::PipelinePrivate);
}

} // namespace

Status supply_residency_cache(PipelineState &pipeline,
                              const VirtualRunProjection &run,
                              const residency::EpochLease lease,
                              const bool scan, Stats &stats) noexcept {
  residency::Pool &pool = *pipeline.residency_pool;
  const UploadResult uploaded =
      upload_cache(pipeline, *pool.cache_input, run, lease.bindings);
  if (!uploaded.status) {
    return uploaded.status;
  }
  std::uint64_t fetched = 0u;
  for (const residency::CacheBinding binding : lease.bindings) {
    fetched += static_cast<std::uint64_t>(binding.fetch);
  }
  const std::uint64_t bytes = fetched * run.input_page_bytes;
  record_pipeline_upload(pipeline, bytes, uploaded);
  Accumulate(stats.buffer_allocations, uploaded.buffer_allocations);
  Accumulate(stats.buffer_reuses, uploaded.buffer_reuses);
  Accumulate(stats.transfer_submissions.host_to_device,
             uploaded.command_submits);
  Accumulate(stats.uploaded_bytes, bytes);
  Accumulate(stats.host_write_bytes, bytes);
  if (scan) {
    std::array<UploadRequest, PipelineLeafCapacity> requests{};
    for (std::size_t index = 0u; index < lease.bindings.size(); ++index) {
      const residency::CacheBinding binding = lease.bindings[index];
      const std::size_t offset =
          static_cast<std::size_t>(binding.frame * run.input_page_bytes);
      requests[index] = UploadRequest{
          .buffer = pool.input.get(),
          .data = run.output_stage + offset,
          .bytes = static_cast<std::size_t>(run.input_page_bytes),
          .offset = offset,
      };
    }
    UploadResult projected{};
    if (pipeline.device->backend == Backend::Cpu) {
      CpuBufferState *const target = cpu_buffer(*pool.input);
      if (target == nullptr || target->data == nullptr) {
        return Status::fail(Reason::TransferInvalid);
      }
      for (std::size_t index = 0u; index < lease.bindings.size(); ++index) {
        const UploadRequest request = requests[index];
        std::memcpy(target->data.get() + request.offset, request.data,
                    request.bytes);
      }
    } else {
      if (pipeline.device->ops == nullptr ||
          pipeline.device->ops->upload_batch == nullptr) {
        return Status::fail(Reason::TransferInvalid);
      }
      projected = pipeline.device->ops->upload_batch(
          *pipeline.device,
          std::span<const UploadRequest>{requests.data(),
                                         lease.bindings.size()},
          node::accel::detail::TransferCompletion::Complete,
          node::accel::detail::TransferAuthority::PipelinePrivate);
      if (!projected.status) {
        return projected.status;
      }
    }
    record_pipeline_upload(
        pipeline, lease.bindings.size() * run.input_page_bytes, projected);
    const std::uint64_t projected_bytes =
        lease.bindings.size() * run.input_page_bytes;
    Accumulate(stats.buffer_allocations, projected.buffer_allocations);
    Accumulate(stats.buffer_reuses, projected.buffer_reuses);
    Accumulate(stats.transfer_submissions.host_to_device,
               projected.command_submits);
    Accumulate(stats.uploaded_bytes, projected_bytes);
    Accumulate(stats.host_write_bytes, projected_bytes);
    return Status::success();
  }
  return copy_frames(pipeline, *pool.cache_input, *pool.input, lease.bindings,
                     run.input_page_bytes, run.input_page_bytes, stats);
}

Status retain_residency_output(PipelineState &pipeline,
                               const VirtualRunProjection &run,
                               const residency::EpochLease lease,
                               Stats &stats) noexcept {
  residency::Pool &pool = *pipeline.residency_pool;
  return copy_frames(pipeline, *pool.output, *pool.cache_output, lease.bindings,
                     run.output_page_bytes, run.output_page_bytes, stats);
}

Status writeback_residency_cache(
    VirtualBacking &output, const VirtualRunProjection &run,
    const std::span<const residency::CacheTransition> transitions,
    ResidencyStats &residency_stats) noexcept {
  std::size_t writeback_count = 0u;
  for (const residency::CacheTransition transition : transitions) {
    writeback_count += static_cast<std::size_t>(
        transition.kind == residency::TransitionKind::Writeback);
  }
  if (writeback_count == 0u) {
    return Status::success();
  }
  // Every execution download publishes an exact host-tier mirror before the
  // corresponding frame becomes Dirty. The device cache remains the physical
  // residency authority; eviction consumes that already-accounted host mirror
  // and therefore performs no duplicate D2H transfer.
  std::array<VirtualWrite, PipelineLeafCapacity> ranges{};
  std::size_t count = 0u;
  std::uint64_t logical_bytes = 0u;
  VirtualBackingAccess::require_recovery(output, run.active.output_bytes);
  for (const residency::CacheTransition transition : transitions) {
    if (transition.kind != residency::TransitionKind::Writeback ||
        transition.frame >= run.frame_capacity ||
        transition.key.page > std::numeric_limits<std::uint64_t>::max() /
                                  run.output_payload_bytes) {
      continue;
    }
    const std::uint64_t logical_offset =
        transition.key.page * run.output_payload_bytes;
    if (logical_offset >= run.active.output_bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t bytes = static_cast<std::size_t>(std::min(
        run.output_payload_bytes, run.active.output_bytes - logical_offset));
    ranges[count++] = VirtualWrite{
        .offset = logical_offset,
        .bytes =
            std::span<const std::byte>{
                run.output_stage + transition.frame * run.output_page_bytes +
                    run.output_prefix_bytes,
                bytes},
    };
    Accumulate(logical_bytes, bytes);
  }
  const std::uint64_t started = pipeline_clock();
  const Status written =
      output.write_batch(std::span<const VirtualWrite>{ranges.data(), count});
  Accumulate(residency_stats.backing_io_ns, pipeline_clock() - started);
  if (!written) {
    return written;
  }
  VirtualBackingAccess::publish_write(output);
  Accumulate(residency_stats.backing_write_bytes, logical_bytes);
  Accumulate(residency_stats.page_out_bytes, logical_bytes);
  Accumulate(residency_stats.page_out_count, count);
  return Status::success();
}

} // namespace rund::compute::detail
