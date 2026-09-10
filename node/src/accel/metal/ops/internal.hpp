#pragma once

#include "../../backend/buffer.hpp"
#include "../../backend/ops/table.hpp"
#include "../ops.hpp"

#include <span>

namespace rund::node::accel::detail {

// The persistent-sliding capability is implemented by the residency owner.
// Keep its BackendOps-facing declaration here so ops.cpp remains composition
// only without importing that owner's implementation details.
[[nodiscard]] PersistentResidencySlidingCapability
QueryMetalPreparedPersistentSlidingCapability(
    std::span<const PreparedResidencyPersistentSlidingRole>, std::uint64_t,
    ResidencySlidingMemory, PersistentResidencySlidingMode) noexcept;

namespace metal_ops_detail {

[[nodiscard]] rund::Buffer Create(const rund::AccelDevice &,
                                  const rund::BufferDesc &,
                                  BackendBufferInitialization,
                                  BackendBufferMemory, std::uint64_t);
[[nodiscard]] std::uint64_t BufferStorageBytes(const rund::AccelDevice &,
                                               std::uint64_t) noexcept;
[[nodiscard]] rund::AccelCheck Upload(const rund::AccelDevice &,
                                      const rund::kernel::ResidentBufferRef &,
                                      const std::shared_ptr<void> &,
                                      const void *, std::uint64_t,
                                      std::uint64_t);
[[nodiscard]] BackendDownload Download(const rund::AccelDevice &,
                                       const rund::kernel::ResidentBufferRef &,
                                       const std::shared_ptr<void> &, void *,
                                       std::uint64_t, std::uint64_t, bool);
[[nodiscard]] BackendUpload UploadBatch(const rund::AccelDevice &,
                                        std::span<const UploadRoute>,
                                        TransferCompletion, TransferAuthority);
[[nodiscard]] BackendDownload DownloadBatch(const rund::AccelDevice &,
                                            std::span<const DownloadRoute>,
                                            TransferAuthority);
[[nodiscard]] BackendCopy CopyBatch(const rund::AccelDevice &,
                                    std::span<const CopyRoute>,
                                    TransferAuthority);
[[nodiscard]] BackendLookup Lookup(const rund::AccelDevice &,
                                   const rund::kernel::ResidentBufferRef &,
                                   const std::shared_ptr<void> &);
[[nodiscard]] BackendHostView HostRead(const rund::AccelDevice &,
                                       const rund::kernel::ResidentBufferRef &,
                                       const std::shared_ptr<void> &) noexcept;
[[nodiscard]] BackendHostWriteView
HostWrite(const rund::AccelDevice &, const rund::kernel::ResidentBufferRef &,
          const std::shared_ptr<void> &) noexcept;

[[nodiscard]] rund::RuntimeStats Stats(const rund::AccelDevice &);
[[nodiscard]] rund::AccelCheck
VirtualPipelineCapability(const rund::AccelDevice &) noexcept;
[[nodiscard]] rund::node::accel::AccelMemoryStats
Memory(const rund::AccelDevice &) noexcept;
[[nodiscard]] bool InjectDeviceLostOnce(const rund::AccelDevice &,
                                        SubmitKind) noexcept;
[[nodiscard]] bool
InjectDownloadFailureOnce(const rund::AccelDevice &) noexcept;
[[nodiscard]] bool
InjectHostReadUnavailableOnce(const rund::AccelDevice &) noexcept;
[[nodiscard]] bool
InjectHostWriteUnavailableOnce(const rund::AccelDevice &) noexcept;
[[nodiscard]] bool
InjectResidencyTerminalLossOnce(const rund::AccelDevice &) noexcept;
[[nodiscard]] bool
InjectTraceUnavailableOnce(const rund::AccelDevice &) noexcept;
[[nodiscard]] bool
InjectTraceResolveDeviceLostOnce(const rund::AccelDevice &) noexcept;

[[nodiscard]] ServiceFreeDirectCapability
ServiceFreeDirectCapabilityFor(const std::shared_ptr<void> &,
                               const ServiceFreeDirectProof &) noexcept;

} // namespace metal_ops_detail
} // namespace rund::node::accel::detail
