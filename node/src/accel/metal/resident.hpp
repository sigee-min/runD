#pragma once

#include "../backend/result.hpp"
#include "resident/model.hpp"
#include <accel/device.hpp>

#include <span>

namespace rund::node::accel::detail {

[[nodiscard]] MetalResidentBufferResult
CreateMetalResidentBuffer(const rund::AccelDevice &pick,
                          const ResidentDesc &desc,
                          bool zero_initialize = false);

[[nodiscard]] MetalResidentBufferResult
LookupMetalResidentBuffer(const rund::AccelDevice &pick,
                          const rund::kernel::ResidentBufferRef &ref,
                          const std::shared_ptr<void> &handle);

[[nodiscard]] rund::AccelCheck
UploadMetalResidentBuffer(const rund::AccelDevice &pick,
                          const rund::kernel::ResidentBufferRef &ref,
                          const std::shared_ptr<void> &handle, const void *data,
                          rund::kernel::u64 bytes, rund::kernel::u64 offset);

[[nodiscard]] BackendUpload
UploadMetalResidentBuffers(const rund::AccelDevice &pick,
                           std::span<const UploadRoute> requests,
                           TransferCompletion completion);

[[nodiscard]] BackendDownload DownloadMetalResidentBuffer(
    const rund::AccelDevice &pick, const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle, void *data, rund::kernel::u64 bytes,
    rund::kernel::u64 offset, bool hash_payload);

[[nodiscard]] BackendDownload
DownloadMetalResidentBuffers(const rund::AccelDevice &pick,
                             std::span<const DownloadRoute> requests);

[[nodiscard]] BackendCopy
CopyMetalResidentBuffers(const rund::AccelDevice &pick,
                         std::span<const CopyRoute> requests);

} // namespace rund::node::accel::detail
