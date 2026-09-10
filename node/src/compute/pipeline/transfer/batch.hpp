#pragma once

#include "../../backend/transfer/model.hpp"
#include "../state.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::compute::detail {

struct PipelineFrameUpload final {
  BufferState *buffer{};
  const void *data{};
  std::size_t bytes{};
  std::size_t offset{};
};

struct PipelineFrameDownload final {
  const BufferState *buffer{};
  void *data{};
  std::size_t bytes{};
  std::size_t offset{};
  std::uint32_t output{};
};

// The exact backend POD is both the publication input and the caller's
// retained evidence. A later Pipeline Stats epoch reset cannot erase the
// physical facts of this transfer.
struct PipelineFrameUploadResult final {
  UploadResult transfer{};
  std::uint64_t bytes{};
};

struct PipelineFrameDownloadResult final {
  DownloadResult transfer{};
  std::uint64_t bytes{};
  std::uint64_t events{};
};

// Source-private prepared-frame boundary. Request and claim storage is fixed by
// PipelineLeafCapacity; the backend sees at most one batch call per operation.
// Callers retain the Pipeline and publication gates, so output observation and
// transfer evidence close under the same authority as Pipeline::read/write.
[[nodiscard]] PipelineFrameUploadResult
upload_pipeline_frames(PipelineState &state,
                       std::span<const PipelineFrameUpload> frames) noexcept;

[[nodiscard]] PipelineFrameDownloadResult download_pipeline_frames(
    PipelineState &state,
    std::span<const PipelineFrameDownload> frames) noexcept;

// Typed VirtualPipeline seam. It accepts only the two canonical residency
// arenas after has_private_residency_authority() proves the complete resource
// set is Pipeline-owned. No Device claim is acquired; ordinary Buffer/Pipeline
// transfers keep using the shared functions above.
[[nodiscard]] PipelineFrameUploadResult upload_pipeline_private_frames(
    PipelineState &state, std::span<const PipelineFrameUpload> frames) noexcept;

[[nodiscard]] PipelineFrameDownloadResult download_pipeline_private_frames(
    PipelineState &state,
    std::span<const PipelineFrameDownload> frames) noexcept;

// Sole output-observation/hash publication owner shared by single read and
// batch download. The caller holds PipelineState::gate and publication::gate.
[[nodiscard]] Status
publish_pipeline_output_observation(PipelineState &state, std::size_t output,
                                    std::uint64_t hash) noexcept;

} // namespace rund::compute::detail
