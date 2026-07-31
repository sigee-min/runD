#pragma once

#include "result.hpp"
#include "token.hpp"

#include <accel/buffer.hpp>
#include <accel/runtime.hpp>

#include <node/accel/buffer.hpp>

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

[[nodiscard]] rund::Buffer
CreateBackendBuffer(const std::shared_ptr<PickToken> &token,
                    const rund::BufferDesc &desc,
                    BackendBufferInitialization initialization =
                        BackendBufferInitialization::Zeroed);

[[nodiscard]] rund::AccelCheck
UploadBackendBuffer(const std::shared_ptr<PickToken> &token,
                    const rund::Buffer &buffer, const void *data,
                    std::uint64_t bytes, std::uint64_t offset);

[[nodiscard]] rund::AccelCheck
UploadBackendBuffer(const std::shared_ptr<PickToken> &token,
                    const rund::kernel::ResidentBufferRef &resident,
                    const std::shared_ptr<void> &handle, const void *data,
                    std::uint64_t bytes, std::uint64_t offset);

[[nodiscard]] BackendUpload
UploadBackendBuffers(const std::shared_ptr<PickToken> &token,
                     std::span<const UploadRoute> requests);

[[nodiscard]] BackendDownload
DownloadBackendBuffer(const std::shared_ptr<PickToken> &token,
                      const rund::Buffer &buffer, void *data,
                      std::uint64_t bytes, std::uint64_t offset,
                      bool hash_payload = false);

[[nodiscard]] BackendDownload
DownloadBackendBuffer(const std::shared_ptr<PickToken> &token,
                      const rund::kernel::ResidentBufferRef &resident,
                      const std::shared_ptr<void> &handle, void *data,
                      std::uint64_t bytes, std::uint64_t offset,
                      bool hash_payload = false);

[[nodiscard]] BackendDownload
DownloadBackendBuffers(const std::shared_ptr<PickToken> &token,
                       std::span<const DownloadRoute> requests);

[[nodiscard]] BackendLookup
LookupBackendBuffer(const std::shared_ptr<PickToken> &token,
                    const rund::kernel::ResidentBufferRef &requested,
                    const std::shared_ptr<void> &handle);

[[nodiscard]] rund::RuntimeStats
ReadBackendStats(const std::shared_ptr<PickToken> &token);

void ResetBackendStats(const std::shared_ptr<PickToken> &token) noexcept;

[[nodiscard]] rund::node::accel::AccelMemoryStats
ReadBackendMemory(const std::shared_ptr<PickToken> &token) noexcept;

} // namespace rund::node::accel::detail
