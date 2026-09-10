#include "../persist.hpp"

#include "../../../../run/projection.hpp"

#include <limits>

namespace rund::compute::detail::graph_reduce::output_persist_detail {

bool project(const VirtualRunProjection &run, const Ticket &ticket,
             Projection &projection) noexcept {
  if (!ticket.output_persist ||
      ticket.batch == std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }
  const auto pages = ticket.output_persist.pages();
  if (pages.empty() || pages.size() != ticket.count) {
    return false;
  }
  projection.count = pages.size();
  projection.token = ticket.batch + 1u;
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    const residency::execution::GraphPersistPage page = pages[index];
    const std::byte *const frame =
        run.host_output_frame_capacity == 0u
            ? virtual_output_frame(run, page.frame)
            : virtual_host_output_frame(run, page.frame);
    if (frame == nullptr || page.bytes == 0u ||
        page.bytes > std::numeric_limits<std::size_t>::max() ||
        page.frame_offset > std::numeric_limits<std::size_t>::max()) {
      return false;
    }
    projection.requests[index] = residency::PersistRequest{
        .key = page.key,
        .backing_offset = page.backing_offset,
        .bytes = static_cast<std::size_t>(page.bytes),
        .frame_offset = static_cast<std::size_t>(page.frame_offset),
        .frame = frame,
        .physical_frame = page.frame,
    };
  }
  return true;
}

} // namespace rund::compute::detail::graph_reduce::output_persist_detail
