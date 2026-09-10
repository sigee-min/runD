#include "../wavefront.hpp"

#include <algorithm>
#include <limits>

namespace rund::compute::detail::graph_reduce {

bool Wavefront::reset(const residency::TiledGraphInvocation &invocation,
                      const residency::TiledGraphPlan &plan) noexcept {
  *this = {};
  if (!invocation.valid() || !invocation.owned_by(plan) ||
      plan.stages().empty() || plan.stages().size() > WavefrontStageCapacity) {
    return false;
  }
  invocation_ = &invocation;
  plan_ = &plan;
  stage_count_ = plan.stages().size();
  return true;
}

bool Wavefront::admit(const std::uint64_t batch) noexcept {
  if (invocation_ == nullptr || plan_ == nullptr ||
      batch >= invocation_->batch_count()) {
    return false;
  }
  const std::size_t bank = static_cast<std::size_t>(batch % WavefrontBankCount);
  for (std::size_t stage = 0u; stage < stage_count_; ++stage) {
    if (cells_[bank * WavefrontStageCapacity + stage].state !=
        CellState::Empty) {
      return false;
    }
  }
  residency::PageRun run{};
  if (!invocation_->batch(batch, run) || run.page_count == 0u) {
    return false;
  }
  for (std::size_t stage = 0u; stage < stage_count_; ++stage) {
    Cell cell{};
    std::size_t predecessor_count = 0u;
    if (!invocation_->predecessors(batch, stage, cell.predecessors,
                                   predecessor_count)) {
      return false;
    }
    cell.predecessor_count = predecessor_count;
    cell.coordinate = WavefrontCoordinate{
        .ordinal = batch * stage_count_ + stage,
        .batch = batch,
        .first_page = run.first_page,
        .page_count = run.page_count,
        .stage = static_cast<std::uint32_t>(stage),
        .resource = std::numeric_limits<std::uint32_t>::max(),
    };
    const residency::TiledGraphStage &node = plan_->stages()[stage];
    for (const residency::TiledGraphPort port : node.ports) {
      cell.coordinate.resource =
          std::min(cell.coordinate.resource, port.resource);
      const residency::TiledGraphResource *const resource =
          plan_->resource(port.resource);
      if (!residency::reads(port.access) || resource == nullptr ||
          resource->kind != residency::GraphResourceKind::ExternalInput ||
          resource->persistence != residency::ResourcePersistence::Backing) {
        continue;
      }
      if (cell.external_input_count >= cell.external_inputs.size() ||
          std::find(cell.external_inputs.begin(),
                    cell.external_inputs.begin() +
                        static_cast<std::ptrdiff_t>(cell.external_input_count),
                    port.resource) !=
              cell.external_inputs.begin() +
                  static_cast<std::ptrdiff_t>(cell.external_input_count)) {
        return false;
      }
      cell.external_inputs[cell.external_input_count] = port.resource;
      cell.required_mask |= std::uint32_t{1u} << static_cast<std::uint32_t>(
                                cell.external_input_count++);
    }
    if (cell.coordinate.resource == std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    cell.device_ready = cell.required_mask == 0u;
    cell.state = CellState::Waiting;
    cells_[bank * WavefrontStageCapacity + stage] = cell;
  }
  return true;
}

} // namespace rund::compute::detail::graph_reduce
