#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

Status
PrefetchController::accept(Ticket &ticket, PrefetchLane &lane,
                           const residency::PrefetchReceipt &receipt) noexcept {
  using prefetch_detail::Accumulate;
  const std::size_t source = ticket.host_ready_count;
  if (source >= ticket.host_ready.size() ||
      ticket.host_supply.page_count > ticket.host_supply.pages.size() ||
      receipt.pages.size() >
          ticket.host_supply.pages.size() - ticket.host_supply.page_count ||
      (source != 0u && ticket.forecast_stage != lane.stage) ||
      std::find(ticket.forecast_resources.begin(),
                ticket.forecast_resources.begin() +
                    static_cast<std::ptrdiff_t>(source),
                lane.resource) != ticket.forecast_resources.begin() +
                                      static_cast<std::ptrdiff_t>(source)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const residency::FrameRegion region = virtual_graph_host_input_region(
      run_, lane.input_index, lane.physical_lane);
  const std::size_t first_page = ticket.host_supply.page_count;
  for (std::size_t index = 0u; index < receipt.pages.size(); ++index) {
    const residency::PrefetchedPage page = receipt.pages[index];
    residency::CacheKey expected{};
    VirtualInputPageProjection projected{};
    if (!residency::project_graph_cache_key(
            lane.materialization, lane.sources[index].key, expected) ||
        page.key != expected || page.frame == nullptr ||
        page.physical_frame < region.first ||
        page.physical_frame - region.first >= region.count ||
        virtual_host_input_frame(run_, page.physical_frame) != page.frame ||
        !project_virtual_input_page(run_, page.key.page, projected) ||
        page.bytes != projected.transfer_bytes ||
        page.target_offset != projected.target_offset) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  residency::execution::GraphReady ready{};
  if (!authority_.graph_forecasts().retire_graph_forecast(
          std::move(lane.forecast), ready)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::destroy_at(&ticket.host_ready[source]);
  std::construct_at(&ticket.host_ready[source], std::move(ready));
  ticket.forecast_stage = lane.stage;
  ticket.forecast_resources[source] = lane.resource;
  ++ticket.host_ready_count;
  ticket.host_supply.speculative =
      ticket.host_supply.speculative || receipt.speculative;
  for (std::size_t index = 0u; index < receipt.pages.size(); ++index) {
    const residency::PrefetchedPage page = receipt.pages[index];
    ticket.host_supply.pages[first_page + index] = page;
    if (page.fetched) {
      ++ticket.host_supply.fetched_pages;
      Accumulate(ticket.host_supply.backing_bytes, page.backing_bytes);
    }
  }
  ticket.host_supply.page_count += receipt.pages.size();
  classify_backing(
      stats_,
      std::span<const residency::PageUse>{lane.sources.data(), lane.count},
      ticket.host_supply.fetched_pages, ticket.host_supply.speculative, true);
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
