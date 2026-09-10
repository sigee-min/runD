#include "internal.hpp"

#include "../../../../pipeline/run/clock.hpp"
#include "../transfer.hpp"

#include <span>

namespace rund::compute::detail::graph_reduce::output_detail {

Download execute(PipelineState &pipeline, Projection &projection) noexcept {
  const std::uint64_t started = pipeline_clock();
  const PipelineFrameDownloadResult downloaded = download_frames(
      pipeline, std::span<const PipelineFrameDownload>{
                    projection.downloads.data(), projection.page_count});
  Download result{
      .status = downloaded.transfer.status,
      .interval = {.started = started, .completed = pipeline_clock()},
      .complete = downloaded.transfer.status &&
                  downloaded.bytes == projection.expected_bytes,
  };
  if (result.complete) {
    result.status = Status::success();
    for (std::size_t index = 0u; index < projection.page_count; ++index) {
      projection.completions[index].bytes = projection.page_bytes[index];
    }
  } else if (result.status) {
    result.status = Status::fail(Reason::CompletionInvalid);
  }
  return result;
}

} // namespace rund::compute::detail::graph_reduce::output_detail
