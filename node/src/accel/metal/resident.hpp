#pragma once

#include "../backend/result.hpp"
#include "resident/model.hpp"
#include <accel/device.hpp>

#include <span>

namespace rund::node::accel::detail {

[[nodiscard]] std::uint64_t
MetalBufferStorageBytes(const rund::AccelDevice &pick,
                        std::uint64_t logical_bytes) noexcept;

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

[[nodiscard]] BackendUpload UploadMetalResidentBuffers(
    const rund::AccelDevice &pick, std::span<const UploadRoute> requests,
    TransferCompletion completion, TransferAuthority authority);

[[nodiscard]] BackendDownload DownloadMetalResidentBuffer(
    const rund::AccelDevice &pick, const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle, void *data, rund::kernel::u64 bytes,
    rund::kernel::u64 offset, bool hash_payload);

[[nodiscard]] BackendDownload
DownloadMetalResidentBuffers(const rund::AccelDevice &pick,
                             std::span<const DownloadRoute> requests,
                             TransferAuthority authority);

// Authenticates the Pipeline-private resident capability and projects the
// Shared MTLBuffer contents as a stable read-only Host view. The caller owns
// the exact native-terminal ordering; this function performs no wait or copy.
[[nodiscard]] BackendHostView
ReadMetalResidentBuffer(const rund::AccelDevice &pick,
                        const rund::kernel::ResidentBufferRef &ref,
                        const std::shared_ptr<void> &handle) noexcept;

// Authenticates the Pipeline-private resident capability and projects the
// Shared MTLBuffer contents as a stable writable Host view. The caller owns
// the exact mutation range and execution ordering.
[[nodiscard]] BackendHostWriteView
WriteMetalResidentBuffer(const rund::AccelDevice &pick,
                         const rund::kernel::ResidentBufferRef &ref,
                         const std::shared_ptr<void> &handle) noexcept;

[[nodiscard]] BackendCopy
CopyMetalResidentBuffers(const rund::AccelDevice &pick,
                         std::span<const CopyRoute> requests,
                         TransferAuthority authority);

} // namespace rund::node::accel::detail
