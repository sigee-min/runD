#include "internal.hpp"

#include <algorithm>
#include <limits>

namespace rund::compute::detail::graph_reduce::promote_detail {

bool project(const VirtualRunProjection &run,
             const residency::EpochLease destination, const Ticket &ticket,
             Projection &result) noexcept {
  result = {};
  const auto promoted = ticket.input_promote.pages();
  if (!ticket.input_promote || promoted.size() > result.completions.size() ||
      ticket.host_supply.page_count > ticket.host_supply.pages.size()) {
    return false;
  }
  const std::span<const residency::PrefetchedPage> supplied{
      ticket.host_supply.pages.data(), ticket.host_supply.page_count};
  for (std::size_t index = 0u; index < promoted.size(); ++index) {
    const residency::execution::GraphPromotePage page = promoted[index];
    const residency::GraphMaterialization *const materialization =
        graph_materialization(run, page.use.resource);
    const auto source =
        std::find_if(supplied.begin(), supplied.end(),
                     [page](const residency::PrefetchedPage value) {
                       return value.key == page.key &&
                              value.physical_frame == page.source_frame;
                     });
    const auto target =
        std::find_if(destination.bindings.begin(), destination.bindings.end(),
                     [page](const residency::CacheBinding value) {
                       return value.key == page.key &&
                              value.frame == page.target_frame && value.fetch;
                     });
    if (materialization == nullptr ||
        page.bytes != materialization->page_bytes ||
        page.bytes > std::numeric_limits<std::size_t>::max() ||
        source == supplied.end() || source->frame == nullptr ||
        source->bytes == 0u || source->bytes > page.bytes ||
        source->target_offset > page.bytes - source->bytes ||
        target == destination.bindings.end()) {
      return false;
    }
    result.promoted[index] = page;
    result.pages[index] = *source;
    result.completions[index] = residency::execution::GraphPromoteCompletion{
        .key = page.key,
        .bytes = page.bytes,
        .source_frame = page.source_frame,
        .target_frame = page.target_frame,
    };
  }
  result.page_count = promoted.size();
  return true;
}

} // namespace rund::compute::detail::graph_reduce::promote_detail
