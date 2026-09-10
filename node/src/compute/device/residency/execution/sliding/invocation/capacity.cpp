#include "../internal.hpp"

namespace rund::compute::detail::residency::execution {

std::size_t SlidingInvocation::scratch_capacity() const noexcept {
  if (topology_ == SlidingTopology::Direct) {
    return direct_ != nullptr && direct_->has_window_footprint()
               ? WindowFootprintSourceCapacity + UseCapacity
               : 2u * UseCapacity;
  }
  if (topology_ != SlidingTopology::Graph || graph_owner_ == nullptr) {
    return 0u;
  }
  const auto &plan = graph_owner_->tiled_graph();
  const std::uint64_t frames =
      std::min(graph_.page_count(), graph_.frame_capacity());
  std::size_t ports = 0u;
  for (const TiledGraphStage &stage : plan.stages()) {
    ports = std::max(ports, stage.ports.size());
  }
  if (frames == 0u || ports == 0u ||
      frames > std::numeric_limits<std::size_t>::max() / ports) {
    return 0u;
  }
  return static_cast<std::size_t>(frames) * ports;
}

bool SlidingInvocation::capacity_requirements(
    std::size_t &inputs, std::size_t &outputs) const noexcept {
  inputs = 0u;
  outputs = 0u;
  if (!*this) {
    return false;
  }
  if (topology_ == SlidingTopology::Direct) {
    inputs = direct_->has_window_footprint()
                 ? static_cast<std::size_t>(std::min<std::uint64_t>(
                       direct_->page_count(), direct_->frame_capacity() + 2u))
                 : direct_->frame_capacity();
    outputs = direct_->frame_capacity();
    return inputs != 0u && inputs <= SlidingHostCapacity &&
           outputs <= SlidingHostCapacity;
  }
  const TiledGraphPlan &plan = graph_owner_->tiled_graph();
  const std::uint64_t active_frames =
      std::min(graph_.page_count(), graph_.frame_capacity());
  if (active_frames == 0u || active_frames > SlidingHostCapacity) {
    return false;
  }
  for (const TiledGraphStage &stage : plan.stages()) {
    std::size_t stage_inputs = 0u;
    std::size_t stage_outputs = 0u;
    for (const TiledGraphPort &port : stage.ports) {
      const TiledGraphResource *const resource = plan.resource(port.resource);
      if (resource == nullptr) {
        return false;
      }
      stage_inputs += static_cast<std::size_t>(
          reads(port.access) &&
          resource->kind == GraphResourceKind::ExternalInput &&
          resource->persistence == ResourcePersistence::Backing);
      stage_outputs += static_cast<std::size_t>(
          writes(port.access) &&
          resource->kind == GraphResourceKind::ExternalOutput);
    }
    const std::size_t frames = static_cast<std::size_t>(active_frames);
    if ((stage_inputs != 0u && frames > SlidingHostCapacity / stage_inputs) ||
        (stage_outputs != 0u && frames > SlidingHostCapacity / stage_outputs)) {
      return false;
    }
    inputs = std::max(inputs, frames * stage_inputs);
    outputs = std::max(outputs, frames * stage_outputs);
  }
  return inputs <= SlidingHostCapacity && outputs <= SlidingHostCapacity;
}

bool SlidingInvocation::host_ring_capacities(
    std::size_t &inputs, std::size_t &outputs) const noexcept {
  inputs = 0u;
  outputs = 0u;
  if (!*this) {
    return false;
  }
  if (topology_ == SlidingTopology::Graph) {
    // The Graph execution owner binds its typed external Host arenas at the
    // Authority join. The topology model only seals per-coordinate demand.
    inputs = SlidingHostCapacity;
    outputs = SlidingHostCapacity;
    return true;
  }
  const auto &input = direct_->host_input_regions();
  const auto &output = direct_->host_output_regions();
  inputs = std::min(input[0].count, input[1].count);
  outputs = std::min(output[0].count, output[1].count);
  return inputs != 0u && outputs != 0u && inputs <= SlidingHostCapacity &&
         outputs <= SlidingHostCapacity;
}

bool SlidingInvocation::host_input_requirement(
    std::size_t &inputs) const noexcept {
  inputs = 0u;
  if (!*this) {
    return false;
  }
  if (topology_ == SlidingTopology::Graph) {
    std::size_t outputs = 0u;
    return capacity_requirements(inputs, outputs);
  }
  if (!direct_->input_sources_materializable()) {
    return false;
  }
  if (direct_->has_window_footprint()) {
    inputs = static_cast<std::size_t>(std::min<std::uint64_t>(
        direct_->page_count(), direct_->frame_capacity() + 2u));
    return inputs != 0u && inputs <= SlidingHostCapacity;
  }
  for (std::size_t bank = 0u; bank < BankCapacity; ++bank) {
    std::uint64_t rows = 0u;
    if (!direct_->input_live_rows(bank, rows) ||
        rows > std::numeric_limits<std::size_t>::max()) {
      return false;
    }
    inputs = std::max(inputs, static_cast<std::size_t>(rows));
  }
  return inputs != 0u && inputs <= SlidingHostCapacity;
}

} // namespace rund::compute::detail::residency::execution
