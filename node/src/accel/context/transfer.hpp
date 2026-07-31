#pragma once

#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>

#include <node/accel/context.hpp>

#include "../backend/result.hpp"

#include <span>

namespace rund::node::accel::detail {

struct UploadEntry final {
  const rund::AccelBuffer *buffer = nullptr;
  const void *data = nullptr;
  std::uint64_t bytes = 0u;
  std::uint64_t offset = 0u;
};

struct DownloadEntry final {
  const rund::AccelBuffer *buffer = nullptr;
  void *data = nullptr;
  std::uint64_t bytes = 0u;
  std::uint64_t offset = 0u;
  std::uint64_t *payload_hash = nullptr;
};

struct AccelTransfer final {
  rund::AccelCheck check{};
  std::uint64_t payload_hash{};
  std::uint64_t staging_bytes{};
  std::uint64_t staging_peak_bytes{};
  std::uint64_t staging_reused_bytes{};
  std::uint64_t staging_budget{};
  std::uint64_t buffer_allocations{};
  std::uint64_t buffer_reuses{};
  std::uint64_t command_submits{};
  std::uint64_t readback_ns{};
  bool staging_reused{};
  bool payload_hash_valid{};
};

[[nodiscard]] AccelTransfer
DownloadAccelBufferMeasured(const rund::AccelContext &context,
                            const rund::AccelBuffer &buffer, void *data,
                            std::uint64_t bytes, std::uint64_t offset = 0u,
                            bool hash_payload = false);

[[nodiscard]] AccelTransfer
UploadAccelBuffers(const rund::AccelContext &context,
                   std::span<const UploadEntry> requests,
                   std::span<UploadRoute> routes);

[[nodiscard]] AccelTransfer
DownloadAccelBuffersMeasured(const rund::AccelContext &context,
                             std::span<const DownloadEntry> requests,
                             std::span<DownloadRoute> routes);

} // namespace rund::node::accel::detail
