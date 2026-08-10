#include "wave.hpp"

#include "../../pipeline/local.hpp"
#include "../stats.hpp"
#include "backing.hpp"

#include <rund/counter.hpp>

#include <mutex>
#include <span>

namespace rund::compute::detail {
namespace {

using ::rund::detail::counter::Accumulate;

[[nodiscard]] PipelineSlotUploadResult
upload_slots(PipelineState &pipeline,
             const std::span<const PipelineSlotUpload> slots) noexcept {
  std::lock_guard pipeline_lock{pipeline.gate};
  std::lock_guard publication_lock{pipeline.publication->gate};
  return upload_pipeline_private_slots(pipeline, slots);
}

[[nodiscard]] PipelineSlotDownloadResult
download_slots(PipelineState &pipeline,
               const std::span<const PipelineSlotDownload> slots) noexcept {
  std::lock_guard pipeline_lock{pipeline.gate};
  std::lock_guard publication_lock{pipeline.publication->gate};
  return download_pipeline_private_slots(pipeline, slots);
}

void accumulate_upload(Stats &stats,
                       const PipelineSlotUploadResult &upload) noexcept {
  Accumulate(stats.buffer_allocations, upload.transfer.buffer_allocations);
  Accumulate(stats.buffer_reuses, upload.transfer.buffer_reuses);
  Accumulate(stats.transfer_submissions.host_to_device,
             upload.transfer.command_submits);
  if (upload.transfer.status) {
    Accumulate(stats.uploaded_bytes, upload.bytes);
    Accumulate(stats.host_write_bytes, upload.bytes);
  }
}

} // namespace

VirtualWaveResult
execute_virtual_wave(VirtualPipelineState &state, VirtualBacking &input,
                     VirtualBacking &output, const VirtualRunProjection &run,
                     const std::uint64_t wave, Stats &stats,
                     ::rund::node::hash_detail::Fnv &output_hash) noexcept {
  VirtualWaveProjection projected{};
  if (!project_virtual_wave(run, wave, projected)) {
    return VirtualWaveResult{.status = Status::fail(Reason::PipelineInvalid),
                             .failed_page = projected.failed_page,
                             .poison_pipeline = true};
  }

  Status status =
      read_virtual_wave(input, projected, run, stats.pipeline.residency);
  if (!status) {
    return VirtualWaveResult{.status = status,
                             .failed_page = projected.failed_page};
  }
  // The backing callback is the byte-traffic authority. Once it succeeds,
  // those bytes were read even if a later resident upload is rejected.
  Accumulate(stats.pipeline.residency.backing_read_bytes,
             projected.logical_input_bytes);

  PipelineState &pipeline = *state.pipeline;
  const PipelineSlotUploadResult uploaded = upload_slots(
      pipeline, std::span<const PipelineSlotUpload>{&run.upload, 1u});
  accumulate_upload(stats, uploaded);
  if (!uploaded.transfer.status || uploaded.bytes != run.input_arena_bytes) {
    return VirtualWaveResult{
        .status = uploaded.transfer.status
                      ? Status::fail(Reason::CompletionInvalid)
                      : uploaded.transfer.status,
        .failed_page = projected.failed_page,
        .poison_pipeline = uploaded.transfer.command_submits != 0u};
  }
  Accumulate(stats.pipeline.residency.load_count, projected.page_count);

  status = run_pipeline_with_mode(
      state.pipeline, node::accel::detail::PipelineSubmitMode::Residency);
  if (!status) {
    const Stats wave_stats = pipeline_stats(state.pipeline);
    const Status folded = accumulate_virtual_wave(stats, wave_stats);
    return VirtualWaveResult{.status = folded ? status : folded,
                             .failed_page = projected.failed_page,
                             .poison_pipeline =
                                 poisoned_pipeline(state.pipeline)};
  }
  Accumulate(stats.pipeline.residency.wave_count, 1u);

  const PipelineSlotDownloadResult downloaded = download_slots(
      pipeline, std::span<const PipelineSlotDownload>{&run.download, 1u});
  const Stats wave_stats = pipeline_stats(state.pipeline);
  const Status folded = accumulate_virtual_wave(stats, wave_stats);
  if (!folded) {
    return VirtualWaveResult{.status = folded,
                             .failed_page = projected.failed_page,
                             .poison_pipeline = true};
  }
  if (!downloaded.transfer.status ||
      downloaded.bytes != run.output_arena_bytes) {
    return VirtualWaveResult{.status =
                                 downloaded.transfer.status
                                     ? Status::fail(Reason::CompletionInvalid)
                                     : downloaded.transfer.status,
                             .failed_page = projected.failed_page};
  }

  output_hash.Bytes(reinterpret_cast<const std::uint8_t *>(run.output_stage),
                    projected.logical_output_bytes);
  status = write_virtual_wave(output, projected, run, stats.pipeline.residency);
  if (!status) {
    return VirtualWaveResult{.status = status,
                             .failed_page = projected.failed_page};
  }
  Accumulate(stats.pipeline.residency.backing_write_bytes,
             projected.logical_output_bytes);
  Accumulate(stats.pipeline.residency.writeback_count, projected.page_count);
  return {};
}

} // namespace rund::compute::detail
