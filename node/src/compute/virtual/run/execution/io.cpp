#include "io.hpp"

#include "../cache.hpp"

#include "../../../device/residency/pool.hpp"
#include "../../../pipeline/execution/attempt.hpp"
#include "../../../pipeline/execution/schedule.hpp"
#include "../../../pipeline/execution/submit.hpp"
#include "../../../pipeline/execution/window.hpp"
#include "../../../pipeline/local.hpp"
#include "../../../pipeline/run/clock.hpp"
#include "../../../pipeline/transfer/batch.hpp"

#include <rund/counter.hpp>

#include <array>
#include <cstring>
#include <mutex>
#include <span>

namespace rund::compute::detail::execution_io {
using ::rund::detail::counter::Accumulate;

namespace {
[[nodiscard]] bool mask_bit(const std::uint32_t mask,
                            const std::size_t local) noexcept {
  return local < 32u && (mask & (std::uint32_t{1u} << local)) != 0u;
}
} // namespace

[[nodiscard]] Status fill_input(VirtualBacking &input,
                                const VirtualRunProjection &run,
                                const residency::ExecutionTicket &ticket,
                                PipelineState *const pipeline,
                                ResidencyStats &stats,
                                std::uint64_t &backing_pages,
                                VirtualInputReuseSeed *const reuse) noexcept {
  backing_pages = 0u;
  if (ticket.phase !=
          static_cast<std::uint8_t>(residency::execution::Phase::Input) ||
      ticket.bindings.empty() || ticket.bindings.size() % 2u != 0u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::size_t count = ticket.bindings.size() / 2u;
  if (count > PipelineLeafCapacity) {
    return Status::fail(Reason::PipelineCapacity);
  }
  std::array<VirtualInputMaterialization, PipelineLeafCapacity> pages{};
  const std::uint32_t direct_fill_mask =
      ticket.coherent_mask & ticket.backing_mask;
  const BufferWriteView coherent = direct_fill_mask != 0u && pipeline != nullptr
                                       ? residency_input_view(*pipeline, run)
                                       : BufferWriteView{};
  if (direct_fill_mask != 0u && !coherent) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (std::size_t local = 0u; local < count; ++local) {
    const residency::CacheBinding binding = ticket.bindings[local];
    const bool fetch = mask_bit(ticket.backing_mask, local);
    const bool direct = mask_bit(direct_fill_mask, local);
    VirtualInputPageProjection page{};
    std::byte *const target =
        direct ? coherent.data + local * run.input_page_bytes
               : virtual_host_input_frame(run, binding.frame);
    if (binding.fetch != fetch || (fetch && target == nullptr) ||
        (direct && (local > coherent.bytes / run.input_page_bytes ||
                    run.input_page_bytes >
                        coherent.bytes - local * run.input_page_bytes)) ||
        !project_virtual_input_page(run, binding.key.page, page) ||
        binding.key != virtual_cache_key(run, run.input_backing,
                                         run.input_version, binding.key.page)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    pages[local] = VirtualInputMaterialization{
        .page = binding.key.page,
        .frame = target,
        .read = fetch,
        .finalize = fetch,
    };
  }
  const std::uint64_t started = pipeline_clock();
  const VirtualInputMaterializationResult materialized =
      materialize_virtual_input(
          input, run,
          std::span<const VirtualInputMaterialization>{pages.data(), count},
          reuse);
  Accumulate(stats.backing_io_ns, pipeline_clock() - started);
  if (!materialized.status) {
    return materialized.status;
  }
  backing_pages = materialized.read_pages;
  Accumulate(stats.backing_read_bytes, materialized.backing_bytes);
  if (reuse != nullptr && count != 0u && pages[count - 1u].frame != nullptr) {
    reuse->page = pages[count - 1u].page;
    reuse->frame = pages[count - 1u].frame;
  }
  return Status::success();
}

[[nodiscard]] Status supply_input(PipelineState &pipeline,
                                  const VirtualRunProjection &run,
                                  const residency::ExecutionTicket &ticket,
                                  Stats &stats,
                                  std::uint64_t &device_pages) noexcept {
  device_pages = 0u;
  const std::size_t count = ticket.bindings.size() / 2u;
  if (count == 0u || count > PipelineLeafCapacity ||
      pipeline.residency_pool == nullptr ||
      pipeline.residency_input >= pipeline.resources.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const PipelineResource &resource =
      pipeline.resources[pipeline.residency_input];
  if (pipeline.residency_bank >= residency::Pool::BankCount) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const residency::FrameRegion device =
      run.input_regions[pipeline.residency_bank];
  if (resource.buffer == nullptr || device.count != run.frame_capacity) {
    return Status::fail(Reason::PipelineInvalid);
  }

  const BufferWriteView coherent = residency_input_view(pipeline, run);
  std::array<PipelineFrameUpload, PipelineLeafCapacity> uploads{};
  std::size_t upload_count = 0u;
  std::uint64_t bytes = 0u;
  for (std::size_t local = 0u; local < count; ++local) {
    const residency::CacheBinding host = ticket.bindings[local];
    const residency::CacheBinding target = ticket.bindings[count + local];
    const bool transfer = mask_bit(ticket.transfer_mask, local);
    const bool direct = mask_bit(ticket.coherent_mask, local);
    if (direct) {
      const bool backing = mask_bit(ticket.backing_mask, local);
      if (transfer || target.fetch != backing || host.fetch != backing ||
          target.frame < device.first || target.frame - device.first != local ||
          host.key != target.key) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (backing) {
        ++device_pages;
        Accumulate(bytes, run.input_page_bytes);
      }
      continue;
    }
    const std::byte *const source = virtual_host_input_frame(run, host.frame);
    if (target.fetch != transfer || source == nullptr ||
        target.frame < device.first || target.frame - device.first != local ||
        host.key != target.key) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (!transfer) {
      continue;
    }
    const std::size_t offset = local * run.input_page_bytes;
    if (coherent) {
      if (offset > coherent.bytes ||
          run.input_page_bytes > coherent.bytes - offset) {
        return Status::fail(Reason::PipelineInvalid);
      }
      std::memcpy(coherent.data + offset, source,
                  static_cast<std::size_t>(run.input_page_bytes));
    } else {
      uploads[upload_count++] = PipelineFrameUpload{
          .buffer = resource.buffer.get(),
          .data = source,
          .bytes = static_cast<std::size_t>(run.input_page_bytes),
          .offset = offset,
      };
    }
    ++device_pages;
    Accumulate(bytes, run.input_page_bytes);
  }
  if (coherent) {
    Accumulate(stats.host_write_bytes, bytes);
    return Status::success();
  }
  if (upload_count == 0u) {
    return Status::success();
  }
  PipelineFrameUploadResult uploaded{};
  {
    std::lock_guard pipeline_lock{pipeline.gate};
    std::lock_guard publication_lock{pipeline.publication->gate};
    uploaded = upload_pipeline_private_frames(
        pipeline,
        std::span<const PipelineFrameUpload>{uploads.data(), upload_count});
  }
  if (!uploaded.transfer.status || uploaded.bytes != bytes) {
    return uploaded.transfer.status ? Status::fail(Reason::CompletionInvalid)
                                    : uploaded.transfer.status;
  }
  Accumulate(stats.buffer_allocations, uploaded.transfer.buffer_allocations);
  Accumulate(stats.buffer_reuses, uploaded.transfer.buffer_reuses);
  Accumulate(stats.transfer_submissions.host_to_device,
             uploaded.transfer.command_submits);
  Accumulate(stats.uploaded_bytes, bytes);
  Accumulate(stats.host_write_bytes, bytes);
  return Status::success();
}

[[nodiscard]] Status collect_output(
    PipelineState &pipeline, const VirtualRunProjection &run,
    const residency::ExecutionTicket &ticket,
    std::span<const std::byte *const> &frames,
    std::array<const std::byte *, PipelineLeafCapacity> &storage) noexcept {
  if (ticket.bindings.empty() || ticket.bindings.size() % 2u != 0u ||
      pipeline.residency_pool == nullptr ||
      pipeline.residency_output >= pipeline.resources.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::size_t count = ticket.bindings.size() / 2u;
  if (count > storage.size()) {
    return Status::fail(Reason::PipelineCapacity);
  }
  const PipelineResource &resource =
      pipeline.resources[pipeline.residency_output];
  if (resource.buffer == nullptr ||
      resource.output == PipelineResource::no_output) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const BufferReadView coherent = residency_output_view(pipeline);
  std::array<PipelineFrameDownload, PipelineLeafCapacity> downloads{};
  std::size_t download_count = 0u;
  for (std::size_t local = 0u; local < count; ++local) {
    const residency::CacheBinding source = ticket.bindings[local * 2u];
    const residency::CacheBinding target = ticket.bindings[local * 2u + 1u];
    std::byte *const host = virtual_host_output_frame(run, target.frame);
    const std::size_t offset = local * run.output_page_bytes;
    if (source.key != target.key || host == nullptr ||
        !mask_bit(ticket.transfer_mask, local)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (coherent) {
      if (offset > coherent.bytes ||
          run.output_page_bytes > coherent.bytes - offset) {
        return Status::fail(Reason::PipelineInvalid);
      }
      storage[local] = coherent.data + offset;
    } else {
      downloads[download_count++] = PipelineFrameDownload{
          .buffer = resource.buffer.get(),
          .data = host,
          .bytes = static_cast<std::size_t>(run.output_page_bytes),
          .offset = offset,
          .output = resource.output,
      };
      storage[local] = host;
    }
  }
  if (!coherent) {
    PipelineFrameDownloadResult downloaded{};
    {
      std::lock_guard pipeline_lock{pipeline.gate};
      std::lock_guard publication_lock{pipeline.publication->gate};
      downloaded = download_pipeline_private_frames(
          pipeline, std::span<const PipelineFrameDownload>{downloads.data(),
                                                           download_count});
    }
    const std::uint64_t expected = count * run.output_page_bytes;
    if (!downloaded.transfer.status || downloaded.bytes != expected) {
      return downloaded.transfer.status
                 ? Status::fail(Reason::CompletionInvalid)
                 : downloaded.transfer.status;
    }
  }
  frames = std::span<const std::byte *const>{storage.data(), count};
  return Status::success();
}

[[nodiscard]] std::uint64_t
failed_page(const residency::ExecutionTicket &ticket) noexcept {
  return ticket.bindings.empty() ? ResidencyStats::no_failed_page
                                 : ticket.bindings.front().key.page;
}

} // namespace rund::compute::detail::execution_io
