#pragma once

#include "../../backend.hpp"
#include <rund/compute/pipeline/shape.hpp>

namespace rund::compute::detail::accel_backend {

[[nodiscard]] UploadResult
upload(DeviceState &device, BufferState &buffer, const void *data,
       std::size_t bytes);

[[nodiscard]] UploadResult upload_batch(
    DeviceState &device, std::span<const UploadRequest> requests,
    node::accel::detail::TransferCompletion completion,
    node::accel::detail::TransferAuthority authority);

[[nodiscard]] DownloadResult
download(DeviceState &device, const BufferState &buffer, void *data,
         std::size_t bytes, std::size_t offset);

[[nodiscard]] DownloadResult download_batch(
    DeviceState &device, std::span<const DownloadRequest> requests,
    node::accel::detail::TransferAuthority authority);

[[nodiscard]] CopyResult copy_batch(
    DeviceState &device, std::span<const CopyRequest> requests,
    node::accel::detail::TransferAuthority authority);

[[nodiscard]] BufferReadView host_read(const DeviceState &device,
                                       const BufferState &buffer) noexcept;

[[nodiscard]] BufferWriteView host_write(const DeviceState &device,
                                         const BufferState &buffer) noexcept;

[[nodiscard]] Status resolve_buffer(const DeviceState &device,
                                    const BufferState &buffer,
                                    std::shared_ptr<void> &handle);

inline constexpr std::size_t transfer_prefix_capacity =
    PipelineTransferCapacity;

} // namespace rund::compute::detail::accel_backend
