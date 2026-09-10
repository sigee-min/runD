#include "../persist.hpp"

#include "../../../../../pipeline/run/clock.hpp"
#include "../../../../run/projection.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>

namespace rund::compute::detail::graph_reduce::output_persist_detail {
namespace {

[[nodiscard]] std::byte *persist_frame(const VirtualRunProjection &run,
                                       const std::uint32_t frame) noexcept {
  return run.host_output_frame_capacity == 0u
             ? virtual_output_frame(run, frame)
             : virtual_host_output_frame(run, frame);
}

} // namespace

PersistIo perform(VirtualBacking &output, const VirtualRunProjection &run,
                  const residency::execution::GraphPersist &persist,
                  ::rund::node::hash_detail::Fnv &hash) noexcept {
  PersistIo result{};
  result.interval.started = pipeline_clock();
  const auto pages = persist.pages();
  result.page_count = pages.size();
  result.status = Status::success();
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    const residency::execution::GraphPersistPage page = pages[index];
    result.completions[index] = residency::execution::GraphPersistCompletion{
        .key = page.key,
        .backing_offset = page.backing_offset,
        .frame = page.frame,
    };
  }
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    const residency::execution::GraphPersistPage page = pages[index];
    std::byte *const frame = persist_frame(run, page.frame);
    std::uint64_t frame_end = 0u;
    std::uint64_t backing_end = 0u;
    if (frame == nullptr || page.bytes == 0u ||
        page.bytes > std::numeric_limits<std::size_t>::max() ||
        !kernel::checked::add(page.frame_offset, page.bytes, frame_end) ||
        frame_end > run.output_page_bytes ||
        !kernel::checked::add(page.backing_offset, page.bytes, backing_end) ||
        backing_end > run.active.output_bytes) {
      result.status = Status::fail(Reason::PipelineInvalid);
      break;
    }
    const std::span<const std::byte> bytes{
        frame + static_cast<std::size_t>(page.frame_offset),
        static_cast<std::size_t>(page.bytes)};
    result.may_write = true;
    result.status = output.write(page.backing_offset, bytes);
    if (!result.status) {
      break;
    }
    result.completions[index].bytes = page.bytes;
    result.bytes += page.bytes;
    ++result.completed_pages;
    hash.Bytes(reinterpret_cast<const std::uint8_t *>(bytes.data()),
               bytes.size());
  }
  result.interval.completed = pipeline_clock();
  return result;
}

bool hash_persisted(const VirtualRunProjection &run,
                    const residency::execution::GraphPersist &persist,
                    ::rund::node::hash_detail::Fnv &hash) noexcept {
  const auto pages = persist.pages();
  for (const residency::execution::GraphPersistPage &page : pages) {
    std::byte *const frame = persist_frame(run, page.frame);
    std::uint64_t frame_end = 0u;
    if (frame == nullptr || page.bytes == 0u ||
        page.bytes > std::numeric_limits<std::size_t>::max() ||
        !kernel::checked::add(page.frame_offset, page.bytes, frame_end) ||
        frame_end > run.output_page_bytes) {
      return false;
    }
    hash.Bytes(reinterpret_cast<const std::uint8_t *>(
                   frame + static_cast<std::size_t>(page.frame_offset)),
               static_cast<std::size_t>(page.bytes));
  }
  return true;
}

} // namespace rund::compute::detail::graph_reduce::output_persist_detail
