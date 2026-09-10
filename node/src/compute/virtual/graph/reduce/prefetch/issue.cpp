#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

Status PrefetchController::issue(const std::size_t lane_index,
                                 PrefetchLane &lane,
                                 const residency::TiledGraphPort input_port,
                                 bool &cleanup_failed) noexcept {
  if (input_port.resource != lane.resource ||
      input_port.access != residency::Access::Read) {
    lane = {};
    return Status::fail(Reason::PipelineInvalid);
  }
  if (lane_index >= residency::Pool::BankCount) {
    lane = {};
    return Status::fail(Reason::PipelineInvalid);
  }
  lane.physical_lane = static_cast<std::uint32_t>(lane_index);
  const residency::FrameRegion region = virtual_graph_host_input_region(
      run_, lane.input_index, lane.physical_lane);
  residency::execution::GraphForecast forecast{};
  const residency::AuthorityResult issued =
      authority_.graph_forecasts().issue_graph_forecast(
          state_.pipeline->residency, run_.active.graph, lane.batch, lane.stage,
          input_port.resource,
          std::span<const residency::PageUse>{lane.sources.data(), lane.count},
          lane.materialization, region, forecast);
  if (!issued) {
    lane = {};
    return authority_status(issued);
  }

  const auto rollback = [&]() noexcept {
    const bool released = reject_graph_prefetch(
        authority_, std::move(forecast), Status::fail(Reason::PipelineInvalid));
    cleanup_failed = !released || cleanup_failed;
    if (!released) {
      // The local forecast remains authenticated when release fails. Move it
      // into this lane and quarantine the lane for cleanup retry; destroying
      // it here would leak the Authority capability and permit slot reuse.
      lane.forecast = std::move(forecast);
      lane.pending = true;
    } else {
      lane = {};
    }
    return Status::fail(released ? Reason::PipelineInvalid
                                 : Reason::PipelineBusy);
  };
  const auto forecast_pages = forecast.pages();
  if (forecast_pages.size() != lane.count ||
      forecast.plan() != state_.pipeline->residency->identity() ||
      forecast.coordinate() != lane.epoch.ordinal) {
    return rollback();
  }

  std::array<residency::PrefetchRequest, PipelineLeafCapacity> requests{};
  std::array<VirtualInputPageProjection, PipelineLeafCapacity>
      projected_pages{};
  for (std::size_t index = 0u; index < lane.count; ++index) {
    const residency::execution::GraphForecastPage binding =
        forecast_pages[index];
    VirtualInputPageProjection &projected = projected_pages[index];
    std::byte *const target = virtual_host_input_frame(run_, binding.frame);
    if (binding.frame < region.first ||
        binding.frame - region.first >= region.count || target == nullptr ||
        !project_virtual_input_page(run_, binding.key.page, projected) ||
        binding.offset != projected.logical_offset ||
        binding.bytes != projected.transfer_bytes) {
      return rollback();
    }
    requests[index] = residency::PrefetchRequest{
        .key = binding.key,
        .offset = binding.offset,
        .bytes = static_cast<std::size_t>(binding.bytes),
        .target_offset = projected.target_offset,
        .read_offset = binding.offset,
        .read_bytes = static_cast<std::size_t>(binding.bytes),
        .read_target_offset = projected.target_offset,
        .frame = target,
        .physical_frame = binding.frame,
        .fetch = binding.fetch,
    };
    VirtualInputReuseProjection reuse{};
    if (index != 0u && binding.fetch &&
        requests[index - 1u].key.page !=
            std::numeric_limits<std::uint64_t>::max() &&
        requests[index - 1u].key.page + 1u == binding.key.page &&
        requests[index - 1u].frame != target &&
        project_virtual_input_reuse(projected_pages[index - 1u], projected,
                                    reuse)) {
      requests[index].read_offset = reuse.logical_offset;
      requests[index].read_bytes = reuse.read_bytes;
      requests[index].read_target_offset = reuse.read_target_offset;
      requests[index].reuse_source_offset = reuse.source_offset;
      requests[index].reuse_target_offset = reuse.target_offset;
      requests[index].reuse_bytes = reuse.bytes;
    }
  }
  if (lane.input_index >= input_count_ ||
      inputs_[lane.input_index] == nullptr ||
      !pool_.prefetch[lane_index].submit(
          *inputs_[lane.input_index],
          std::span<const residency::PrefetchRequest>{requests.data(),
                                                      lane.count},
          forecast.token(), lane.speculative)) {
    return rollback();
  }
  lane.forecast = std::move(forecast);
  lane.pending = true;
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
