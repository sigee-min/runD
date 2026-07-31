#pragma once

#include <accel/check.hpp>

#include <kernel/program/compute/binding/model.hpp>

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

struct UploadRoute final {
  rund::kernel::ResidentBufferRef resident{};
  std::shared_ptr<void> handle{};
  const void *data = nullptr;
  std::uint64_t bytes = 0u;
  std::uint64_t offset = 0u;
};

struct DownloadRoute final {
  rund::kernel::ResidentBufferRef resident{};
  std::shared_ptr<void> handle{};
  void *data = nullptr;
  std::uint64_t bytes = 0u;
  std::uint64_t offset = 0u;
  std::uint64_t *payload_hash = nullptr;
};

struct BackendUpload final {
  rund::AccelCheck check{false, "accel_buffer_backend_unavailable"};
  std::uint64_t staging_bytes = 0u;
  std::uint64_t staging_peak_bytes = 0u;
  std::uint64_t staging_reused_bytes = 0u;
  std::uint64_t buffer_allocations = 0u;
  std::uint64_t buffer_reuses = 0u;
  std::uint64_t command_submits = 0u;
};

struct BackendDownload final {
  rund::AccelCheck check{false, "accel_buffer_backend_unavailable"};
  std::uint64_t payload_hash = 0u;
  std::uint64_t staging_bytes = 0u;
  std::uint64_t staging_peak_bytes = 0u;
  std::uint64_t staging_reused_bytes = 0u;
  std::uint64_t buffer_allocations = 0u;
  std::uint64_t buffer_reuses = 0u;
  std::uint64_t command_submits = 0u;
  bool staging_reused = false;
  bool payload_hash_valid = false;
};

struct BackendLookup final {
  rund::AccelCheck check{false, "accel_context_buffer_invalid"};
  rund::kernel::ResidentBufferRef ref{};
  std::shared_ptr<void> handle{};
};

} // namespace rund::node::accel::detail
