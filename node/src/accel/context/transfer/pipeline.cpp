#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>

#include "../../backend/resource.hpp"
#include "../../kernel/prepared/interface/api.hpp"
#include "../local.hpp"
#include "../transfer.hpp"

#include <cstdint>
#include <memory>

namespace rund::node::accel {

rund::AccelCheck detail::PrepareAccelPipelineTransfer(
    const rund::AccelContext &context, const PreparedKernelPipeline &prepared,
    const rund::AccelBuffer &input, const rund::AccelBuffer &output,
    const std::uint64_t input_bytes, const std::uint64_t output_bytes,
    const std::uint64_t exact_storage_bytes) {
  const TransferAdmission source = AdmitAccelBufferTransfer(context, input);
  const TransferAdmission target = AdmitAccelBufferTransfer(context, output);
  if (!source.check.ok) {
    return source.check;
  }
  if (!target.check.ok) {
    return target.check;
  }
  if (input_bytes == 0u || output_bytes == 0u || exact_storage_bytes == 0u ||
      input_bytes > source.byte_extent || output_bytes > target.byte_extent) {
    return RejectAccelCheck("accel_buffer_transfer_overflow");
  }
  const UploadRoute upload{.resident = source.route.ref,
                           .handle = source.route.handle,
                           .bytes = input_bytes};
  const DownloadRoute download{.resident = target.route.ref,
                               .handle = target.route.handle,
                               .bytes = output_bytes};
  return PreparePreparedKernelPipelineTransfer(prepared, upload, download,
                                               exact_storage_bytes);
}

detail::AccelTransfer detail::UploadPreparedAccelPipeline(
    const rund::AccelContext &context, const PreparedKernelPipeline &prepared,
    const void *const data, const std::uint64_t bytes) noexcept {
  const BackendUpload uploaded =
      UploadPreparedKernelPipeline(prepared, data, bytes);
  return {.check =
              TransferCheckFrom(uploaded.check, "accel_buffer_upload_overflow"),
          .staging_bytes = uploaded.staging_bytes,
          .staging_peak_bytes = uploaded.staging_peak_bytes,
          .staging_reused_bytes = uploaded.staging_reused_bytes,
          .staging_budget = context.pick.caps.staging_bytes,
          .buffer_allocations = uploaded.buffer_allocations,
          .buffer_reuses = uploaded.buffer_reuses,
          .command_submits = uploaded.command_submits,
          .staging_reused =
              uploaded.staging_bytes != 0u &&
              uploaded.staging_reused_bytes == uploaded.staging_bytes};
}

detail::AccelTransfer detail::DownloadPreparedAccelPipeline(
    const rund::AccelContext &context, const PreparedKernelPipeline &prepared,
    void *const data, const std::uint64_t bytes,
    std::uint64_t *const payload_hash) noexcept {
  const BackendDownload downloaded =
      DownloadPreparedKernelPipeline(prepared, data, bytes, payload_hash);
  return {.check = TransferCheckFrom(downloaded.check,
                                     "accel_buffer_download_overflow"),
          .payload_hash = downloaded.payload_hash,
          .staging_bytes = downloaded.staging_bytes,
          .staging_peak_bytes = downloaded.staging_peak_bytes,
          .staging_reused_bytes = downloaded.staging_reused_bytes,
          .staging_budget = context.pick.caps.staging_bytes,
          .buffer_allocations = downloaded.buffer_allocations,
          .buffer_reuses = downloaded.buffer_reuses,
          .command_submits = downloaded.command_submits,
          .readback_ns = downloaded.readback_ns,
          .confirmed_bytes = downloaded.confirmed_bytes,
          .ordered_prefix = downloaded.ordered_prefix,
          .first_failed = downloaded.first_failed,
          .staging_reused = downloaded.staging_reused,
          .payload_hash_valid =
              downloaded.check.ok && downloaded.payload_hash_valid,
          .first_failed_valid = downloaded.first_failed_valid};
}

} // namespace rund::node::accel
