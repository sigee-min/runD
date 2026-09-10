#include "../batch.hpp"

#include "../fill.hpp"

namespace rund::compute::detail::sliding_product_detail {
namespace {

[[nodiscard]] bool prepare_entry(SlidingProductRun &state, FetchBatch &batch,
                                 FetchBatchEntry &entry,
                                 std::uint64_t &bytes) noexcept {
  entry.source = entry.fetch.source();
  entry.frame = virtual_host_input_frame(*state.run, entry.fetch.frame());
  if (entry.frame == nullptr || !entry.source.materializes_frame() ||
      entry.source.frame_bytes > std::numeric_limits<std::size_t>::max() ||
      !prepare_fetch_backing_slice(state, entry.fetch, entry.frame,
                                   entry.slice) ||
      entry.slice.target_offset > entry.source.frame_bytes ||
      entry.slice.bytes >
          entry.source.frame_bytes - entry.slice.target_offset ||
      entry.slice.bytes > std::numeric_limits<std::size_t>::max() ||
      entry.slice.bytes > std::numeric_limits<std::uint64_t>::max() - bytes) {
    return false;
  }
  bytes += entry.slice.bytes;
  if (entry.slice.bytes != 0u) {
    batch.ranges[batch.range_count++] = VirtualRead{
        .offset = entry.slice.offset,
        .bytes =
            std::span<std::byte>{entry.frame + entry.slice.target_offset,
                                 static_cast<std::size_t>(entry.slice.bytes)},
    };
  }
  return true;
}

} // namespace

FetchBatchIo read_fetch_batch(SlidingProductRun &state,
                              FetchBatch &batch) noexcept {
  FetchBatchIo result{};
  result.io.started_ns = pipeline_clock();
  std::uint64_t bytes = 0u;
  for (std::size_t index = 0u; index < batch.count; ++index) {
    FetchBatchEntry &entry = batch.entries[index];
    if (!entry.fetch.requires_backing()) {
      continue;
    }
    result.backing = true;
    result.may_write = true;
    if (state.run == nullptr || !prepare_entry(state, batch, entry, bytes)) {
      result.io.status = Status::fail(Reason::PipelineInvalid);
      return result;
    }
  }
  if (batch.range_count != 0u) {
    if (state.input == nullptr) {
      result.io.status = Status::fail(Reason::PipelineInvalid);
      return result;
    }
    result.io.status = state.input->read_batch(
        std::span<const VirtualRead>{batch.ranges.data(), batch.range_count});
    if (!result.io.status) {
      return result;
    }
  } else {
    result.io.status = Status::success();
  }
  for (std::size_t index = 0u; index < batch.count; ++index) {
    const FetchBatchEntry &entry = batch.entries[index];
    if (entry.fetch.requires_backing() &&
        !fill_fetch_frame(entry.frame, entry.source)) {
      result.io.status = Status::fail(Reason::PipelineInvalid);
      return result;
    }
  }
  result.io.bytes = bytes;
  return result;
}

} // namespace rund::compute::detail::sliding_product_detail
