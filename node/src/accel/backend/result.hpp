#pragma once

#include <accel/check.hpp>

#include <kernel/program/compute/binding/model.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

// Node Compute bounds one prepared transfer batch to 64 routes.  Keep that
// common warm path entirely inline without making 64 a generic Accel limit;
// larger callers retain the overflow path owned by each backend.
inline constexpr std::size_t kInlineTransferCapacity = 64u;

enum class TransferCompletion : std::uint8_t {
  // Submission may return after the backend has retained every source and
  // target owner needed by its queued command.
  Queued,
  // Success is observable only after every command in the batch completes.
  Complete,
};

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

struct CopyRoute final {
  rund::kernel::ResidentBufferRef source{};
  std::shared_ptr<void> source_handle{};
  rund::kernel::ResidentBufferRef target{};
  std::shared_ptr<void> target_handle{};
  std::uint64_t bytes = 0u;
  std::uint64_t source_offset = 0u;
  std::uint64_t target_offset = 0u;
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

struct BackendCopy final {
  rund::AccelCheck check{false, "accel_buffer_backend_unavailable"};
  std::uint64_t command_submits = 0u;
};

struct BackendLookup final {
  rund::AccelCheck check{false, "accel_context_buffer_invalid"};
  rund::kernel::ResidentBufferRef ref{};
  std::shared_ptr<void> handle{};
};

} // namespace rund::node::accel::detail
