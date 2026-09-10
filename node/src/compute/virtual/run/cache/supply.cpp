#include "../../../backend.hpp"
#include "internal.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

namespace rund::compute::detail {
namespace {

using ::rund::detail::counter::Accumulate;

[[nodiscard]] bool input_bank_frame(const PipelineState &pipeline,
                                    const VirtualRunProjection &run,
                                    const std::uint32_t frame,
                                    std::size_t &local) noexcept {
  if (pipeline.residency_bank >= residency::Pool::BankCount) {
    return false;
  }
  const residency::FrameRegion region =
      pipeline.residency_pool->input_regions[pipeline.residency_bank];
  if (region.count != run.frame_capacity || frame < region.first ||
      frame - region.first >= region.count) {
    return false;
  }
  local = static_cast<std::size_t>(frame - region.first);
  return true;
}

[[nodiscard]] UploadResult
upload_bank(PipelineState &pipeline, BufferState &buffer,
            const VirtualRunProjection &run,
            const std::span<const residency::CacheBinding> bindings,
            const residency::PrefetchReceipt *const host) noexcept {
  std::array<UploadRequest, PipelineLeafCapacity> requests{};
  std::size_t count = 0u;
  for (const residency::CacheBinding &binding : bindings) {
    if (!binding.fetch) {
      continue;
    }
    std::size_t local = 0u;
    if (!input_bank_frame(pipeline, run, binding.frame, local) ||
        local >
            std::numeric_limits<std::size_t>::max() / run.input_page_bytes) {
      return UploadResult{.status = Status::fail(Reason::PipelineInvalid)};
    }
    const std::size_t offset =
        static_cast<std::size_t>(local * run.input_page_bytes);
    std::byte *source = nullptr;
    if (pipeline.device->backend == Backend::Cpu) {
      source = virtual_input_frame(run, binding.frame);
    } else if (host != nullptr) {
      const auto page =
          std::find_if(host->pages.begin(), host->pages.end(),
                       [binding](const residency::PrefetchedPage value) {
                         return value.key == binding.key;
                       });
      source = page == host->pages.end() || page->frame == nullptr ||
                       virtual_host_input_frame(run, page->physical_frame) !=
                           page->frame
                   ? nullptr
                   : page->frame;
    }
    if (source == nullptr) {
      return UploadResult{.status = Status::fail(Reason::PipelineInvalid)};
    }
    requests[count++] = UploadRequest{
        .buffer = &buffer,
        .data = source,
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

Status
supply_residency_cache(PipelineState &pipeline, const VirtualRunProjection &run,
                       const residency::EpochLease lease, Stats &stats,
                       const residency::PrefetchReceipt *const host,
                       VirtualTransferInterval *const interval) noexcept {
  residency::Pool &pool = *pipeline.residency_pool;
  if (pipeline.residency_bank >= residency::Pool::BankCount ||
      pool.input[pipeline.residency_bank] == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (pipeline.device->backend == Backend::Cpu) {
    for (const residency::CacheBinding &binding : lease.bindings) {
      if (virtual_input_frame(run, binding.frame) == nullptr) {
        return Status::fail(Reason::PipelineInvalid);
      }
    }
    return Status::success();
  }
  if (host != nullptr && host->coherent_input) {
    const BufferWriteView view = residency_input_view(pipeline, run);
    if (!view || host->token == 0u || host->token != lease.token) {
      return Status::fail(Reason::PipelineInvalid);
    }
    std::uint64_t pages = 0u;
    for (const residency::CacheBinding &binding : lease.bindings) {
      std::size_t local = 0u;
      const auto found =
          std::find_if(host->pages.begin(), host->pages.end(),
                       [binding](const residency::PrefetchedPage page) {
                         return page.key == binding.key;
                       });
      if (!input_bank_frame(pipeline, run, binding.frame, local) ||
          local >
              std::numeric_limits<std::size_t>::max() / run.input_page_bytes ||
          found == host->pages.end() ||
          found->physical_frame != binding.frame ||
          found->frame != view.data + local * run.input_page_bytes ||
          found->fetched != binding.fetch) {
        return Status::fail(Reason::PipelineInvalid);
      }
      pages += static_cast<std::uint64_t>(binding.fetch);
    }
    Accumulate(stats.host_write_bytes, pages * run.input_page_bytes);
    return Status::success();
  }
  if (host != nullptr && host->coherent_deferred) {
    // The Host receipt was created while this Device bank's previous
    // generation was still live. Admission above proves that generation has
    // now terminated, so promote the exact Host pages through the stable
    // HostVisible owner without manufacturing a transfer submission.
    const BufferWriteView view = residency_input_view(pipeline, run);
    if (!view) {
      return Status::fail(Reason::PipelineInvalid);
    }
    std::uint64_t pages = 0u;
    for (const residency::CacheBinding &binding : lease.bindings) {
      if (!binding.fetch) {
        continue;
      }
      std::size_t local = 0u;
      const auto found =
          std::find_if(host->pages.begin(), host->pages.end(),
                       [binding](const residency::PrefetchedPage page) {
                         return page.key == binding.key;
                       });
      if (!input_bank_frame(pipeline, run, binding.frame, local) ||
          local >
              std::numeric_limits<std::size_t>::max() / run.input_page_bytes ||
          found == host->pages.end() || found->frame == nullptr ||
          virtual_host_input_frame(run, found->physical_frame) !=
              found->frame) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const std::size_t offset = local * run.input_page_bytes;
      if (offset > view.bytes || run.input_page_bytes > view.bytes - offset) {
        return Status::fail(Reason::PipelineInvalid);
      }
      std::memcpy(view.data + offset, found->frame,
                  static_cast<std::size_t>(run.input_page_bytes));
      ++pages;
    }
    Accumulate(stats.host_write_bytes, pages * run.input_page_bytes);
    return Status::success();
  }
  const std::uint64_t started = pipeline_clock();
  const UploadResult uploaded =
      upload_bank(pipeline, *pool.input[pipeline.residency_bank], run,
                  lease.bindings, host);
  const std::uint64_t completed = pipeline_clock();
  if (interval != nullptr) {
    *interval = VirtualTransferInterval{.started_ns = started,
                                        .completed_ns = completed};
  }
  if (!uploaded.status) {
    return uploaded.status;
  }
  std::uint64_t pages = 0u;
  for (const residency::CacheBinding &binding : lease.bindings) {
    pages += static_cast<std::uint64_t>(binding.fetch);
  }
  const std::uint64_t bytes = pages * run.input_page_bytes;
  record_pipeline_upload(pipeline, bytes, uploaded);
  Accumulate(stats.buffer_allocations, uploaded.buffer_allocations);
  Accumulate(stats.buffer_reuses, uploaded.buffer_reuses);
  Accumulate(stats.transfer_submissions.host_to_device,
             uploaded.command_submits);
  Accumulate(stats.uploaded_bytes, bytes);
  Accumulate(stats.host_write_bytes, bytes);
  return Status::success();
}

} // namespace rund::compute::detail
