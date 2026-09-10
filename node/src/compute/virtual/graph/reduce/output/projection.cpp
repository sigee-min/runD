#include "internal.hpp"

#include <kernel/core/checked.hpp>

#include <limits>

namespace rund::compute::detail::graph_reduce::output_detail {

bool project(const VirtualRunProjection &run, const Ticket &ticket,
             Projection &projection) noexcept {
  if (!ticket.output_drain || ticket.collective == nullptr ||
      ticket.collective->residency_output >=
          ticket.collective->resources.size()) {
    return false;
  }
  const PipelineResource &resource =
      ticket.collective->resources[ticket.collective->residency_output];
  const auto pages = ticket.output_drain.pages();
  if (pages.size() != ticket.count) {
    return false;
  }
  projection.page_count = pages.size();
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    const residency::execution::GraphDrainPage page = pages[index];
    projection.completions[index] = residency::execution::GraphDrainCompletion{
        .key = page.key,
        .bytes = 0u,
        .source_frame = page.source_frame,
        .target_frame = page.target_frame,
    };
    std::byte *const target = virtual_host_output_frame(run, page.target_frame);
    std::uint64_t accumulated = 0u;
    std::uint64_t offset = 0u;
    if (target == nullptr || page.source_frame < ticket.output_region.first ||
        page.source_frame - ticket.output_region.first >=
            ticket.output_region.count ||
        page.target_frame != ticket.resident_outputs[index].frame ||
        page.bytes > std::numeric_limits<std::size_t>::max() ||
        !kernel::checked::add(projection.expected_bytes, page.bytes,
                              accumulated) ||
        !kernel::checked::mul(page.source_frame - ticket.output_region.first,
                              run.output_page_bytes, offset) ||
        offset > std::numeric_limits<std::size_t>::max()) {
      return false;
    }
    projection.expected_bytes = accumulated;
    projection.downloads[index] = PipelineFrameDownload{
        .buffer = resource.buffer.get(),
        .data = target,
        .bytes = static_cast<std::size_t>(page.bytes),
        .offset = static_cast<std::size_t>(offset),
        .output = resource.output,
    };
    projection.page_bytes[index] = page.bytes;
  }
  return true;
}

} // namespace rund::compute::detail::graph_reduce::output_detail
