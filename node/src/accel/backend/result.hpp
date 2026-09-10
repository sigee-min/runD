#pragma once

#include <accel/check.hpp>

#include <kernel/program/compute/binding/model.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

enum class BackendBufferMemory : std::uint8_t {
  DeviceLocal,
  HostVisiblePreferred,
};

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

// Identifies the concurrency authority already held by the caller. Shared
// transfers retain backend-wide validation/accounting. PipelinePrivate is
// accepted only after Compute has proved the resident routes are sealed,
// Pipeline-owned resources under that Pipeline's owner gate.
enum class TransferAuthority : std::uint8_t {
  Shared,
  PipelinePrivate,
};

// Per-range completion is distinct from the aggregate transfer status.  A
// failed range may have written an unknown prefix, while every later range is
// still untouched and therefore cannot be treated as valid output.
enum class DownloadRangeState : std::uint8_t {
  Untouched,
  Complete,
  FailedNoWrite,
  FailedMayWrite,
};

struct DownloadRangeOutcome final {
  DownloadRangeState state = DownloadRangeState::Untouched;
  std::uint64_t confirmed_bytes = 0u;
  std::uint64_t payload_hash = 0u;
  bool hash_valid = false;
};

inline void ResetDownloadOutcome(DownloadRangeOutcome *const outcome) noexcept {
  if (outcome != nullptr) {
    *outcome = DownloadRangeOutcome{};
  }
}

inline void MarkDownloadComplete(DownloadRangeOutcome *const outcome,
                                 const std::uint64_t bytes,
                                 const std::uint64_t payload_hash,
                                 const bool hash_valid) noexcept {
  if (outcome != nullptr) {
    *outcome = DownloadRangeOutcome{
        .state = DownloadRangeState::Complete,
        .confirmed_bytes = bytes,
        .payload_hash = payload_hash,
        .hash_valid = hash_valid,
    };
  }
}

inline void
MarkDownloadFailure(DownloadRangeOutcome *const outcome,
                    const DownloadRangeState state,
                    const std::uint64_t confirmed_bytes = 0u) noexcept {
  if (outcome != nullptr) {
    *outcome = DownloadRangeOutcome{
        .state = state,
        .confirmed_bytes = confirmed_bytes,
    };
  }
}

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
  DownloadRangeOutcome *outcome = nullptr;
  const char *failure_reason = nullptr;
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
  std::uint64_t readback_ns = 0u;
  std::uint64_t confirmed_bytes = 0u;
  std::uint64_t ordered_prefix = 0u;
  std::uint64_t first_failed = 0u;
  bool staging_reused = false;
  bool payload_hash_valid = false;
  bool first_failed_valid = false;
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

// A backend may expose resident storage as one stable read-only Host view.
// The retained Buffer capability owns the lifetime; callers must still wait
// for the exact native execution terminal before reading produced bytes.
// Unsupported or non-coherent memory returns the default empty view.
struct BackendHostView final {
  const std::byte *data = nullptr;
  std::uint64_t bytes = 0u;

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return data != nullptr && bytes != 0u;
  }
};

// A backend may expose resident storage as one stable writable Host view.
// The capability authenticates the complete resident Buffer only; callers
// still need their own exact range and mutation authority before writing.
// Unsupported or non-coherent memory returns the default empty view.
struct BackendHostWriteView final {
  std::byte *data = nullptr;
  std::uint64_t bytes = 0u;

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return data != nullptr && bytes != 0u;
  }
};

} // namespace rund::node::accel::detail
