#pragma once

#include "../../backend/result.hpp"

#include "../../backend/buffer.hpp"
#include "../../backend/ops/table.hpp"
#include "../../kernel/residency/service_free_direct.hpp"
#include "../ops.hpp"

namespace rund::node::accel::detail::vulkan_ops_detail {

[[nodiscard]] rund::Buffer Create(
    const rund::AccelDevice &, const rund::BufferDesc &,
    BackendBufferInitialization, BackendBufferMemory, std::uint64_t);
[[nodiscard]] std::uint64_t BufferStorageBytes(const rund::AccelDevice &,
                                               std::uint64_t) noexcept;
[[nodiscard]] std::uint64_t
PipelineTransferStorageBytes(const rund::AccelDevice &, std::uint64_t) noexcept;
[[nodiscard]] rund::AccelCheck Upload(
    const rund::AccelDevice &, const rund::kernel::ResidentBufferRef &,
    const std::shared_ptr<void> &, const void *, std::uint64_t, std::uint64_t);
[[nodiscard]] BackendDownload Download(
    const rund::AccelDevice &, const rund::kernel::ResidentBufferRef &,
    const std::shared_ptr<void> &, void *, std::uint64_t, std::uint64_t, bool);
[[nodiscard]] BackendUpload UploadBatch(const rund::AccelDevice &,
                                        std::span<const UploadRoute>,
                                        TransferCompletion, TransferAuthority);
[[nodiscard]] BackendDownload DownloadBatch(const rund::AccelDevice &,
                                             std::span<const DownloadRoute>,
                                             TransferAuthority);
[[nodiscard]] BackendCopy CopyBatch(const rund::AccelDevice &,
                                    std::span<const CopyRoute>,
                                    TransferAuthority);
[[nodiscard]] BackendLookup Lookup(
    const rund::AccelDevice &, const rund::kernel::ResidentBufferRef &,
    const std::shared_ptr<void> &);
[[nodiscard]] BackendHostView HostRead(
    const rund::AccelDevice &, const rund::kernel::ResidentBufferRef &,
    const std::shared_ptr<void> &) noexcept;
[[nodiscard]] BackendHostWriteView HostWrite(
    const rund::AccelDevice &, const rund::kernel::ResidentBufferRef &,
    const std::shared_ptr<void> &) noexcept;

[[nodiscard]] rund::RuntimeStats Stats(const rund::AccelDevice &);
void Reset(const rund::AccelDevice &);
[[nodiscard]] rund::AccelCheck
VirtualPipelineCapability(const rund::AccelDevice &) noexcept;
[[nodiscard]] rund::node::accel::AccelMemoryStats
Memory(const rund::AccelDevice &) noexcept;
[[nodiscard]] bool InjectDeviceLostOnce(const rund::AccelDevice &,
                                        SubmitKind) noexcept;
[[nodiscard]] bool
InjectHostReadUnavailableOnce(const rund::AccelDevice &) noexcept;
[[nodiscard]] bool
InjectHostWriteUnavailableOnce(const rund::AccelDevice &) noexcept;
[[nodiscard]] bool
InjectResidencyTerminalLossOnce(const rund::AccelDevice &) noexcept;
[[nodiscard]] bool InjectTraceUnavailableOnce(const rund::AccelDevice &) noexcept;

[[nodiscard]] ServiceFreeDirectCapability
ServiceFreeDirectCapabilityFor(const std::shared_ptr<void> &,
                               const ServiceFreeDirectProof &) noexcept;

} // namespace rund::node::accel::detail::vulkan_ops_detail
