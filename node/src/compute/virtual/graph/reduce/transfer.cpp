#include "transfer.hpp"
#include "../../../backend.hpp"

#include "../../../pipeline/local.hpp"
#include "../../../pipeline/run/clock.hpp"
#include "../../stats.hpp"

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>
#include <rund/compute/pipeline/runtime.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <mutex>

namespace rund::compute::detail::graph_reduce {

using ::rund::detail::counter::Accumulate;

Status fold_stage(Stats &stats, const std::shared_ptr<PipelineState> &pipeline,
                  const std::uint64_t identity) noexcept {
  return accumulate_virtual_graph_stage(stats, pipeline_stats(pipeline),
                                        identity);
}

PipelineFrameDownloadResult
download_frames(PipelineState &pipeline,
                const std::span<const PipelineFrameDownload> frames) noexcept {
  std::lock_guard pipeline_lock{pipeline.gate};
  std::lock_guard publication_lock{pipeline.publication->gate};
  return download_pipeline_private_frames(pipeline, frames);
}

Status write_controls(PipelineState &pipeline, const VirtualRunProjection &run,
                      const std::span<const residency::CacheBinding> bindings,
                      const residency::FrameRegion anchor_region, Stats &stats,
                      VirtualTransferInterval *const interval) noexcept {
  residency::Pool *const pool = pipeline.residency_pool.get();
  if (pool == nullptr ||
      pipeline.residency_bank >= residency::Pool::BankCount ||
      bindings.empty() || bindings.size() > PipelineLeafCapacity ||
      pool->control[pipeline.residency_bank] == nullptr ||
      anchor_region.count != run.frame_capacity) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::uint32_t bank = pipeline.residency_bank;
  const std::uint32_t first = anchor_region.first;
  std::array<std::uint64_t, PipelineLeafCapacity> values{};
  std::array<UploadRequest, PipelineLeafCapacity> requests{};
  std::array<bool, PipelineLeafCapacity> local_used{};
  for (std::size_t ordinal = 0u; ordinal < bindings.size(); ++ordinal) {
    const residency::CacheBinding binding = bindings[ordinal];
    if (binding.frame < first || binding.frame >= first + run.frame_capacity ||
        binding.key.page >= run.active.graph.page_count()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t local = binding.frame - first;
    if (local >= PipelineLeafCapacity || local_used[local]) {
      return Status::fail(Reason::PipelineInvalid);
    }
    local_used[local] = true;
    std::uint64_t logical_first = 0u;
    if (!kernel::checked::mul(binding.key.page, run.input_frame_elements,
                              logical_first) ||
        logical_first >= run.active.active_count) {
      return Status::fail(Reason::PipelineInvalid);
    }
    values[local] = std::min(run.input_frame_elements,
                             run.active.active_count - logical_first);
    requests[ordinal] = UploadRequest{
        .buffer = pool->control[bank].get(),
        .data = &values[local],
        .bytes = sizeof(std::uint64_t),
        .offset = local * sizeof(std::uint64_t),
    };
  }
  if (pipeline.device->backend == Backend::Cpu) {
    std::byte *const base = run.control_host_banks[bank];
    if (base == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    for (std::size_t local = 0u; local < local_used.size(); ++local) {
      if (local_used[local]) {
        std::memcpy(base + local * sizeof(std::uint64_t), &values[local],
                    sizeof(std::uint64_t));
      }
    }
    return Status::success();
  }
  if (pipeline.device->ops == nullptr ||
      pipeline.device->ops->upload_batch == nullptr) {
    return Status::fail(Reason::TransferInvalid);
  }
  const std::uint64_t started = pipeline_clock();
  const UploadResult uploaded = pipeline.device->ops->upload_batch(
      *pipeline.device,
      std::span<const UploadRequest>{requests.data(), bindings.size()},
      node::accel::detail::TransferCompletion::Complete,
      node::accel::detail::TransferAuthority::PipelinePrivate);
  const std::uint64_t completed = pipeline_clock();
  if (interval != nullptr) {
    *interval = VirtualTransferInterval{.started_ns = started,
                                        .completed_ns = completed};
  }
  record_pipeline_upload(pipeline, bindings.size() * sizeof(std::uint64_t),
                         uploaded);
  Accumulate(stats.buffer_allocations, uploaded.buffer_allocations);
  Accumulate(stats.buffer_reuses, uploaded.buffer_reuses);
  Accumulate(stats.transfer_submissions.host_to_device,
             uploaded.command_submits);
  if (uploaded.status) {
    Accumulate(stats.uploaded_bytes, bindings.size() * sizeof(std::uint64_t));
    Accumulate(stats.host_write_bytes, bindings.size() * sizeof(std::uint64_t));
  }
  return uploaded.status;
}

Status relocate_graph_lease(PipelineState &pipeline,
                            const residency::TiledGraphPlan &graph,
                            residency::Pool &pool,
                            const residency::EpochLease lease,
                            Stats &stats) noexcept {
  if (lease.relocations.empty()) {
    return Status::success();
  }
  if (pipeline.device == nullptr || lease.ports.empty()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (const residency::GraphRelocation move : lease.relocations) {
    if (move.port >= lease.ports.size() ||
        move.source_frame == move.target_frame) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const residency::GraphLeasePort &port = lease.ports[move.port];
    const residency::TiledGraphResource *const resource =
        graph.resource(port.resource);
    const residency::PoolPhysicalOwner *const owner =
        resource == nullptr ? nullptr : pool.graph_owner(resource->physical_id);
    if (owner == nullptr || owner->arena == nullptr ||
        move.bytes != resource->page_bytes ||
        port.cache_region_count != residency::Pool::BankCount ||
        port.cache_regions != owner->cache_regions) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const auto locate = [owner, resource](const std::uint32_t frame,
                                          std::shared_ptr<BufferState> &buffer,
                                          std::size_t &offset) noexcept {
      for (std::size_t bank = 0u; bank < residency::Pool::BankCount; ++bank) {
        const residency::FrameRegion region = owner->cache_regions[bank];
        if (frame < region.first || frame - region.first >= region.count ||
            owner->buffers[bank] == nullptr) {
          continue;
        }
        std::uint64_t exact_offset = 0u;
        if (!kernel::checked::mul(frame - region.first, resource->page_bytes,
                                  exact_offset) ||
            exact_offset > std::numeric_limits<std::size_t>::max()) {
          return false;
        }
        buffer = owner->buffers[bank];
        offset = static_cast<std::size_t>(exact_offset);
        return true;
      }
      return false;
    };
    std::shared_ptr<BufferState> source;
    std::shared_ptr<BufferState> target;
    std::size_t source_offset = 0u;
    std::size_t target_offset = 0u;
    if (!locate(move.source_frame, source, source_offset) ||
        !locate(move.target_frame, target, target_offset) ||
        source == nullptr || target == nullptr ||
        move.bytes > std::numeric_limits<std::size_t>::max()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t bytes = static_cast<std::size_t>(move.bytes);
    if (pipeline.device->backend == Backend::Cpu) {
      const CpuBufferState *const source_cpu = cpu_buffer(*source);
      CpuBufferState *const target_cpu = cpu_buffer(*target);
      if (source_cpu == nullptr || target_cpu == nullptr ||
          source_cpu->data == nullptr || target_cpu->data == nullptr ||
          source_offset > source_cpu->bytes ||
          bytes > source_cpu->bytes - source_offset ||
          target_offset > target_cpu->bytes ||
          bytes > target_cpu->bytes - target_offset) {
        return Status::fail(Reason::TransferInvalid);
      }
      std::memmove(target_cpu->data.get() + target_offset,
                   source_cpu->data.get() + source_offset, bytes);
      continue;
    }
    if (pipeline.device->ops == nullptr ||
        pipeline.device->ops->copy_batch == nullptr) {
      return Status::fail(Reason::TransferInvalid);
    }
    const CopyRequest request{.source = source.get(),
                              .target = target.get(),
                              .bytes = bytes,
                              .source_offset = source_offset,
                              .target_offset = target_offset};
    const CopyResult copied = pipeline.device->ops->copy_batch(
        *pipeline.device, std::span<const CopyRequest>{&request, 1u},
        node::accel::detail::TransferAuthority::PipelinePrivate);
    Accumulate(stats.transfer_submissions.device_to_device,
               copied.command_submits);
    if (!copied.status) {
      return copied.status;
    }
    record_transfer(*pipeline.device, move.bytes);
  }
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
