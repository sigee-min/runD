#pragma once

#include "src/compute/backend.hpp"
#include "src/compute/pipeline/transfer/batch.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund_node_test_pipeline_transfer {

using rund::compute::detail::BufferState;
using rund::compute::detail::PipelineFrameDownload;
using rund::compute::detail::PipelineFrameDownloadResult;
using rund::compute::detail::PipelineFrameUpload;
using rund::compute::detail::PipelineFrameUploadResult;
using rund::compute::detail::PipelineState;

struct NativeBuffer final {
  std::array<std::byte, 32u> bytes{};
};

struct TransferProbe final {
  std::uint64_t upload_calls{};
  std::uint64_t download_calls{};
  std::size_t upload_routes{};
  std::size_t download_routes{};
  bool fail_upload{};
  bool fail_download_once{};
};

extern TransferProbe *active_probe;

[[nodiscard]] NativeBuffer *native(BufferState &buffer) noexcept;
[[nodiscard]] const NativeBuffer *native(const BufferState &buffer) noexcept;

struct Fixture final {
  static constexpr std::size_t first_bytes = 16u;
  static constexpr std::size_t tail_bytes = 12u;
  static constexpr std::size_t total_bytes = first_bytes + tail_bytes;

  std::shared_ptr<rund::compute::detail::DeviceState> device;
  std::shared_ptr<PipelineState> state;
  std::array<std::shared_ptr<BufferState>, 2u> inputs{};
  std::array<std::shared_ptr<BufferState>, 2u> outputs{};

  Fixture();

  [[nodiscard]] std::shared_ptr<BufferState>
  make_buffer(std::size_t bytes, std::uint64_t generation) const;
  void fill_output(std::size_t index, std::byte seed) const;
};

[[nodiscard]] PipelineFrameUploadResult
transfer_locked(Fixture &fixture, std::span<const PipelineFrameUpload> frames);
[[nodiscard]] PipelineFrameDownloadResult
transfer_locked(Fixture &fixture,
                std::span<const PipelineFrameDownload> frames);

[[nodiscard]] std::array<PipelineFrameUpload, 2u>
uploads(Fixture &fixture, const std::array<std::uint32_t, 4u> &first,
        const std::array<std::uint32_t, 3u> &tail) noexcept;

} // namespace rund_node_test_pipeline_transfer
