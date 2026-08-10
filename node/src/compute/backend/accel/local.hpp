#pragma once

#include "../../backend.hpp"

namespace rund::compute::detail::accel_backend {

[[nodiscard]] Status allocate_buffer(DeviceState &device, BufferState &buffer,
                                     std::size_t scalar_bytes,
                                     std::size_t count, bool zero_initialize,
                                     std::uint64_t exact_storage_bytes);

[[nodiscard]] std::uint64_t
buffer_storage_bytes(const DeviceState &device,
                     std::uint64_t logical_bytes) noexcept;

[[nodiscard]] std::uint64_t
pipeline_transfer_storage_bytes(const DeviceState &device,
                                std::uint64_t logical_bytes) noexcept;

[[nodiscard]] Status
prepare_pipeline_transfer(PipelineState &pipeline) noexcept;

[[nodiscard]] UploadResult upload_pipeline_transfer(PipelineState &pipeline,
                                                    const void *data,
                                                    std::size_t bytes) noexcept;

[[nodiscard]] DownloadResult
download_pipeline_transfer(PipelineState &pipeline, void *data,
                           std::size_t bytes,
                           std::uint64_t *payload_hash) noexcept;

[[nodiscard]] Status
prepare_pipeline_residency(PipelineState &pipeline) noexcept;

} // namespace rund::compute::detail::accel_backend
