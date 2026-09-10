#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

Status PrefetchController::consume(Ticket &ticket,
                                   Timeline *const hidden_by) noexcept {
  if (input_count_ == 0u || graph_.stages().empty()) {
    return Status::fail(Reason::PipelineInvalid);
  }

  std::array<std::uint32_t, residency::execution::GraphPromoteSourceCapacity>
      resources{};
  std::size_t required = 0u;
  const std::span<const residency::TiledGraphStage> stages = graph_.stages();
  for (const residency::TiledGraphPort port : stages.front().ports) {
    const residency::TiledGraphResource *const resource =
        graph_.resource(port.resource);
    if (!residency::reads(port.access) || resource == nullptr ||
        resource->kind != residency::GraphResourceKind::ExternalInput ||
        resource->persistence != residency::ResourcePersistence::Backing) {
      continue;
    }
    std::size_t input = 0u;
    if (required >= resources.size() || !find_input(port.resource, input) ||
        std::find(resources.begin(),
                  resources.begin() + static_cast<std::ptrdiff_t>(required),
                  port.resource) !=
            resources.begin() + static_cast<std::ptrdiff_t>(required)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    resources[required++] = port.resource;
    bool cleanup_failed = false;
    Status status = schedule_input(ticket.batch, input, false, cleanup_failed);
    if (status && !cleanup_failed) {
      status = consume(ticket, 0u, hidden_by);
    }
    if (!status || cleanup_failed) {
      return !status ? status : Status::fail(Reason::PipelineBusy);
    }
  }
  if (required == 0u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  ticket.phase = TicketPhase::SupplyReady;
  return Status::success();
}

Status PrefetchController::consume(Ticket &ticket, const std::uint32_t stage,
                                   Timeline *const hidden_by) noexcept {
  const std::size_t selected_index = lane_index(ticket.batch);
  PrefetchLane &selected = lanes_[selected_index];
  if (stage == 0u && !selected.pending) {
    bool cleanup_failed = false;
    const Status scheduled = schedule(ticket.batch, false, cleanup_failed);
    if (!scheduled || cleanup_failed) {
      return !scheduled ? scheduled : Status::fail(Reason::PipelineBusy);
    }
  }
  if (!selected.pending || selected.batch != ticket.batch ||
      selected.stage != stage || selected.count > ticket.count) {
    return Status::fail(Reason::PipelineInvalid);
  }
  return consume_lane(ticket, selected_index, hidden_by);
}

Status PrefetchController::consume_lane(Ticket &ticket,
                                        const std::size_t selected_index,
                                        Timeline *const hidden_by) noexcept {
  if (selected_index >= lanes_.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  PrefetchLane &selected = lanes_[selected_index];
  if (!selected.pending || selected.batch != ticket.batch ||
      selected.count > ticket.count ||
      selected.physical_lane >= residency::Pool::BankCount ||
      selected.physical_lane != selected_index) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (selected.device_only) {
    if (selected.count != 0u) {
      selected = {};
      return Status::fail(Reason::PipelineInvalid);
    }
    const bool ready =
        wavefront_.host_ready(ticket.batch, selected.stage, selected.resource);
    selected = {};
    return ready ? Status::success() : Status::fail(Reason::PipelineInvalid);
  }

  const std::uint64_t started = pipeline_clock();
  const residency::PrefetchReceipt receipt =
      pool_.prefetch[selected.physical_lane].wait();
  const Interval wait{.started = started, .completed = pipeline_clock()};
  selected.pending = false;
  const Status forecast_terminal =
      terminal_graph_prefetch(authority_, selected.forecast, receipt);
  const bool wavefront_terminal =
      forecast_terminal &&
      wavefront_.forecast_terminal(ticket.batch, selected.stage,
                                   selected.resource, receipt.status);
  const bool recorded =
      record_interval(hidden_by, wait, std::nullopt, stats_.pipeline.residency);
  observe(receipt);

  const bool exact =
      receipt.status && receipt.pages.size() == selected.count &&
      receipt.speculative == selected.speculative &&
      ticket.pages.first_page == selected.pages.first_page &&
      ticket.pages.page_count == selected.pages.page_count &&
      receipt.token == selected.forecast.token() &&
      selected.forecast.coordinate() == selected.epoch.ordinal;
  if (!forecast_terminal || !wavefront_terminal || !recorded || !exact) {
    const Status status = !forecast_terminal
                              ? forecast_terminal
                              : Status::fail(Reason::PipelineInvalid);
    if (!forecast_terminal) {
      const WavefrontCoordinate coordinate{
          .ordinal = selected.epoch.ordinal,
          .batch = selected.batch,
          .first_page = selected.pages.first_page,
          .page_count = selected.pages.page_count,
          .stage = selected.stage,
          .resource = selected.resource};
      (void)wavefront_.cancel_forecast(coordinate, selected.resource);
    }
    const bool settled = settle(selected);
    if (!settled) {
      // A failed release leaves the terminalled (or still live) capability in
      // the lane. Keep it quarantined so a later batch cannot overwrite the
      // forecast before abort/cleanup retries its authenticated release.
      selected.pending = false;
      return Status::fail(Reason::PipelineBusy);
    }
    selected = {};
    return settled ? status : Status::fail(Reason::PipelineBusy);
  }

  const Status accepted = accept(ticket, selected, receipt);
  if (!accepted) {
    const bool settled = settle(selected);
    if (!settled) {
      selected.pending = false;
      return Status::fail(Reason::PipelineBusy);
    }
  }
  selected = {};
  return accepted;
}

Status PrefetchController::poll(Ticket &ticket, bool &progressed) noexcept {
  progressed = false;
  for (std::size_t index = 0u; index < lanes_.size(); ++index) {
    PrefetchLane &selected = lanes_[index];
    if (!selected.pending || selected.batch != ticket.batch ||
        selected.device_only || selected.stage == 0u ||
        selected.stage + 1u >= graph_.stages().size() ||
        (ticket.host_ready_count != 0u &&
         ticket.forecast_stage != selected.stage) ||
        !pool_.prefetch[index].ready()) {
      continue;
    }
    const Status consumed = consume_lane(ticket, index, nullptr);
    if (!consumed) {
      return consumed;
    }
    progressed = true;
    break;
  }
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
