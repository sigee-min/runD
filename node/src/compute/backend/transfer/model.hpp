#pragma once

#include "../../../accel/backend/result.hpp"
#include <rund/compute/status.hpp>

#include <cstddef>
#include <cstdint>

namespace rund::compute::detail {

struct BufferState;

struct DownloadResult final {
  Status status{Status::success()};
  std::uint64_t payload_hash{};
  std::uint64_t staging_bytes{};
  std::uint64_t staging_peak_bytes{};
  std::uint64_t staging_reused_bytes{};
  std::uint64_t staging_budget{};
  std::uint64_t buffer_allocations{};
  std::uint64_t buffer_reuses{};
  std::uint64_t command_submits{};
  std::uint64_t readback_ns{};
  std::uint64_t confirmed_bytes{};
  std::uint64_t ordered_prefix{};
  std::uint64_t first_failed{};
  bool staging_reused{};
  bool payload_hash_valid{};
  bool first_failed_valid{};
};

struct UploadResult final {
  Status status{Status::success()};
  std::uint64_t staging_bytes{};
  std::uint64_t staging_peak_bytes{};
  std::uint64_t staging_reused_bytes{};
  std::uint64_t staging_budget{};
  std::uint64_t buffer_allocations{};
  std::uint64_t buffer_reuses{};
  std::uint64_t command_submits{};
};

struct CopyResult final {
  Status status{Status::success()};
  std::uint64_t command_submits{};
};

struct BufferReadView final {
  const std::byte *data{};
  std::size_t bytes{};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return data != nullptr && bytes != 0u;
  }
};

struct BufferWriteView final {
  std::byte *data{};
  std::size_t bytes{};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return data != nullptr && bytes != 0u;
  }
};

struct UploadRequest final {
  BufferState *buffer = nullptr;
  const void *data = nullptr;
  std::size_t bytes = 0u;
  std::size_t offset = 0u;
};

struct DownloadRequest final {
  const BufferState *buffer = nullptr;
  void *data = nullptr;
  std::size_t bytes = 0u;
  std::size_t offset = 0u;
  std::uint64_t *payload_hash = nullptr;
  node::accel::detail::DownloadRangeOutcome *outcome = nullptr;
};

struct CopyRequest final {
  const BufferState *source = nullptr;
  BufferState *target = nullptr;
  std::size_t bytes = 0u;
  std::size_t source_offset = 0u;
  std::size_t target_offset = 0u;
};

} // namespace rund::compute::detail
