#include "resource.hpp"

#include "match.hpp"
#include "usage.hpp"

#include <rund/counter.hpp>

#include <algorithm>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] bool
ValidRoute(const std::shared_ptr<PickToken> &token) noexcept {
  return token != nullptr && token->ops != nullptr && token->raw.check.ok &&
         token->raw.owner != nullptr && token->raw.api == token->ops->api;
}

[[nodiscard]] rund::AccelCheck
Validate(const std::shared_ptr<PickToken> &token, const rund::Buffer &buffer,
         const void *const data, const std::uint64_t bytes,
         const std::uint64_t offset,
         const char *const overflow_reason) noexcept {
  if (!ValidRoute(token) || !buffer.check.ok || buffer.id == 0u ||
      buffer.bytes == 0u || buffer.owner == nullptr ||
      buffer.handle == nullptr || !KnownUsage(buffer.usage) ||
      (bytes != 0u && data == nullptr)) {
    return rund::AccelCheck{false, "accel_buffer_unavailable"};
  }
  if (buffer.owner.get() != static_cast<const void *>(token.get())) {
    return rund::AccelCheck{false, "accel_buffer_owner_mismatch"};
  }
  if (!SameOwner(buffer.owner, token)) {
    return rund::AccelCheck{false, "accel_buffer_backend_unavailable"};
  }
  if (offset > buffer.bytes || bytes > buffer.bytes - offset) {
    return rund::AccelCheck{false, overflow_reason};
  }
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] rund::kernel::ResidentBufferRef
Resident(const rund::Buffer &buffer) noexcept {
  return rund::kernel::ResidentBufferRef{
      .id = buffer.id,
      .bytes = buffer.bytes,
      .element_bytes = buffer.element_bytes,
      .stride_bytes = buffer.stride_bytes,
      .count = buffer.count,
      .usage = ResidentUsage(buffer.usage),
  };
}

} // namespace

rund::Buffer
CreateBackendBuffer(const std::shared_ptr<PickToken> &token,
                    const rund::BufferDesc &desc,
                    const BackendBufferInitialization initialization) {
  if (!ValidRoute(token) || token->ops->create == nullptr) {
    return rund::Buffer{
        .check = rund::AccelCheck{false, "accel_buffer_backend_unavailable"}};
  }
  rund::Buffer created = token->ops->create(token->raw, desc, initialization);
  if (!created.check.ok) {
    return created;
  }
  if (created.id == 0u || created.bytes == 0u || created.handle == nullptr ||
      !SameObject(created.owner, token->raw.owner)) {
    return rund::Buffer{
        .check = rund::AccelCheck{false, "accel_buffer_backend_unavailable"}};
  }
  created.owner = PublicPickOwner(token);
  return created;
}

rund::AccelCheck UploadBackendBuffer(const std::shared_ptr<PickToken> &token,
                                     const rund::Buffer &buffer,
                                     const void *const data,
                                     const std::uint64_t bytes,
                                     const std::uint64_t offset) {
  const rund::AccelCheck check = Validate(token, buffer, data, bytes, offset,
                                          "accel_buffer_upload_overflow");
  if (!check.ok || token->ops->upload == nullptr) {
    return check.ok
               ? rund::AccelCheck{false, "accel_buffer_backend_unavailable"}
               : check;
  }
  return UploadBackendBuffer(token, Resident(buffer), buffer.handle, data,
                             bytes, offset);
}

rund::AccelCheck
UploadBackendBuffer(const std::shared_ptr<PickToken> &token,
                    const rund::kernel::ResidentBufferRef &resident,
                    const std::shared_ptr<void> &handle, const void *const data,
                    const std::uint64_t bytes, const std::uint64_t offset) {
  if (!ValidRoute(token) || token->ops->upload == nullptr ||
      resident.id == 0u || resident.bytes == 0u || handle == nullptr ||
      (bytes != 0u && data == nullptr) || offset > resident.bytes ||
      bytes > resident.bytes - offset) {
    return rund::AccelCheck{false, "accel_buffer_backend_unavailable"};
  }
  return token->ops->upload(token->raw, resident, handle, data, bytes, offset);
}

BackendUpload
UploadBackendBuffers(const std::shared_ptr<PickToken> &token,
                     const std::span<const UploadRoute> requests) {
  if (!ValidRoute(token) || requests.empty()) {
    return {};
  }
  if (token->ops->upload_batch != nullptr) {
    return token->ops->upload_batch(token->raw, requests);
  }
  if (token->ops->upload == nullptr) {
    return {};
  }
  for (const UploadRoute &request : requests) {
    const rund::AccelCheck uploaded =
        token->ops->upload(token->raw, request.resident, request.handle,
                           request.data, request.bytes, request.offset);
    if (!uploaded.ok) {
      return BackendUpload{.check = uploaded};
    }
  }
  return BackendUpload{.check = {true, "ok"}};
}

BackendDownload DownloadBackendBuffer(const std::shared_ptr<PickToken> &token,
                                      const rund::Buffer &buffer,
                                      void *const data,
                                      const std::uint64_t bytes,
                                      const std::uint64_t offset,
                                      const bool hash_payload) {
  const rund::AccelCheck check = Validate(token, buffer, data, bytes, offset,
                                          "accel_buffer_download_overflow");
  if (!check.ok || token->ops->download == nullptr) {
    return BackendDownload{
        .check = check.ok ? rund::AccelCheck{false,
                                             "accel_buffer_backend_unavailable"}
                          : check};
  }
  return DownloadBackendBuffer(token, Resident(buffer), buffer.handle, data,
                               bytes, offset, hash_payload);
}

BackendDownload
DownloadBackendBuffer(const std::shared_ptr<PickToken> &token,
                      const rund::kernel::ResidentBufferRef &resident,
                      const std::shared_ptr<void> &handle, void *const data,
                      const std::uint64_t bytes, const std::uint64_t offset,
                      const bool hash_payload) {
  if (!ValidRoute(token) || token->ops->download == nullptr ||
      resident.id == 0u || resident.bytes == 0u || handle == nullptr ||
      (bytes != 0u && data == nullptr) || offset > resident.bytes ||
      bytes > resident.bytes - offset) {
    return {};
  }
  return token->ops->download(token->raw, resident, handle, data, bytes, offset,
                              hash_payload);
}

BackendDownload
DownloadBackendBuffers(const std::shared_ptr<PickToken> &token,
                       const std::span<const DownloadRoute> requests) {
  if (!ValidRoute(token) || requests.empty()) {
    return {};
  }
  if (token->ops->download_batch != nullptr) {
    return token->ops->download_batch(token->raw, requests);
  }
  if (token->ops->download == nullptr) {
    return {};
  }
  BackendDownload total{
      .check = rund::AccelCheck{true, "ok"},
      .payload_hash_valid = true,
  };
  for (const DownloadRoute &request : requests) {
    const BackendDownload downloaded =
        token->ops->download(token->raw, request.resident, request.handle,
                             request.data, request.bytes, request.offset, true);
    if (!downloaded.check.ok || !downloaded.payload_hash_valid) {
      return downloaded;
    }
    *request.payload_hash = downloaded.payload_hash;
    total.staging_bytes = ::rund::detail::counter::SaturatingAdd(
        total.staging_bytes, downloaded.staging_bytes);
    total.staging_reused_bytes = ::rund::detail::counter::SaturatingAdd(
        total.staging_reused_bytes,
        downloaded.staging_reused_bytes != 0u ? downloaded.staging_reused_bytes
        : downloaded.staging_reused           ? downloaded.staging_bytes
                                              : 0u);
    total.staging_peak_bytes =
        std::max(total.staging_peak_bytes, downloaded.staging_peak_bytes != 0u
                                               ? downloaded.staging_peak_bytes
                                               : downloaded.staging_bytes);
    total.buffer_allocations = ::rund::detail::counter::SaturatingAdd(
        total.buffer_allocations, downloaded.buffer_allocations);
    total.buffer_reuses = ::rund::detail::counter::SaturatingAdd(
        total.buffer_reuses, downloaded.buffer_reuses);
    total.command_submits = ::rund::detail::counter::SaturatingAdd(
        total.command_submits, downloaded.command_submits);
    total.staging_reused = total.staging_reused_bytes == total.staging_bytes &&
                           total.staging_bytes != 0u;
  }
  return total;
}

BackendLookup
LookupBackendBuffer(const std::shared_ptr<PickToken> &token,
                    const rund::kernel::ResidentBufferRef &requested,
                    const std::shared_ptr<void> &handle) {
  if (!ValidRoute(token) || token->ops->lookup == nullptr) {
    return {};
  }
  return token->ops->lookup(token->raw, requested, handle);
}

rund::RuntimeStats ReadBackendStats(const std::shared_ptr<PickToken> &token) {
  if (!ValidRoute(token) || token->ops->stats == nullptr) {
    return rund::RuntimeStats{.reason = "accel_buffer_backend_unavailable"};
  }
  return token->ops->stats(token->raw);
}

void ResetBackendStats(const std::shared_ptr<PickToken> &token) noexcept {
  if (ValidRoute(token) && token->ops->reset != nullptr) {
    token->ops->reset(token->raw);
  }
}

rund::node::accel::AccelMemoryStats
ReadBackendMemory(const std::shared_ptr<PickToken> &token) noexcept {
  if (!ValidRoute(token) || token->ops->memory == nullptr) {
    return {};
  }
  return token->ops->memory(token->raw);
}

} // namespace rund::node::accel::detail
