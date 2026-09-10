#include "../wavefront.hpp"

namespace rund::compute::detail::graph_reduce {

bool Wavefront::forecast_terminal(const std::uint64_t batch,
                                  const std::uint32_t stage,
                                  const std::uint32_t resource,
                                  const Status status) noexcept {
  Cell *const cell = find(batch, stage);
  if (cell == nullptr || cell->state != CellState::Waiting ||
      cell->device_ready) {
    return false;
  }
  for (std::size_t index = 0u; index < cell->external_input_count; ++index) {
    if (cell->external_inputs[index] != resource) {
      continue;
    }
    const std::uint32_t bit = std::uint32_t{1u}
                              << static_cast<std::uint32_t>(index);
    if ((cell->host_ready_mask & bit) != 0u) {
      return false;
    }
    cell->forecast_mask &= ~bit;
    if (!status) {
      cell->failed = true;
      return true;
    }
    return host_ready(batch, stage, resource);
  }
  return false;
}

bool Wavefront::host_ready(const std::uint64_t batch, const std::uint32_t stage,
                           const std::uint32_t resource) noexcept {
  Cell *const cell = find(batch, stage);
  if (cell == nullptr || cell->state != CellState::Waiting || cell->failed ||
      cell->device_ready) {
    return false;
  }
  for (std::size_t index = 0u; index < cell->external_input_count; ++index) {
    if (cell->external_inputs[index] != resource) {
      continue;
    }
    const std::uint32_t bit = std::uint32_t{1u}
                              << static_cast<std::uint32_t>(index);
    if ((cell->host_ready_mask & bit) != 0u) {
      return false;
    }
    cell->forecast_mask &= ~bit;
    cell->host_ready_mask |= bit;
    return true;
  }
  return false;
}

bool Wavefront::device_ready(const std::uint64_t batch,
                             const std::uint32_t stage) noexcept {
  Cell *const cell = find(batch, stage);
  if (cell == nullptr || cell->state != CellState::Waiting || cell->failed ||
      cell->device_ready ||
      (cell->required_mask != 0u &&
       cell->host_ready_mask != cell->required_mask)) {
    return false;
  }
  cell->device_ready = true;
  return true;
}

bool Wavefront::device_resident(const std::uint64_t batch,
                                const std::uint32_t stage) noexcept {
  Cell *const cell = find(batch, stage);
  if (cell == nullptr || cell->state != CellState::Waiting || cell->failed ||
      cell->device_ready) {
    return false;
  }
  cell->host_ready_mask = cell->required_mask;
  cell->device_ready = true;
  return true;
}

bool Wavefront::already_ready(const std::uint64_t batch,
                              const std::uint32_t stage) const noexcept {
  const Cell *const cell = find(batch, stage);
  return cell != nullptr && cell->state == CellState::Waiting &&
         !cell->failed && cell->device_ready;
}

bool Wavefront::dependency_ready(const std::uint64_t batch,
                                 const std::uint32_t stage) noexcept {
  Cell *const cell = find(batch, stage);
  if (cell == nullptr || cell->state != CellState::Waiting || cell->failed ||
      cell->device_ready || cell->required_mask != 0u) {
    return false;
  }
  cell->device_ready = true;
  return true;
}

} // namespace rund::compute::detail::graph_reduce
