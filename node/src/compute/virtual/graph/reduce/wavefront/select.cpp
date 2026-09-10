#include "../wavefront.hpp"

namespace rund::compute::detail::graph_reduce {

bool Wavefront::predecessors_ready(const Cell &cell) const noexcept {
  for (std::size_t index = 0u; index < cell.predecessor_count; ++index) {
    const residency::TiledGraphDependency dependency = cell.predecessors[index];
    if (dependency.stage >= stage_count_ ||
        dependency.phase >
            residency::TiledGraphDependencyPhase::ReleaseComplete) {
      return false;
    }
    const std::size_t bank =
        static_cast<std::size_t>(dependency.batch % WavefrontBankCount);
    if (dependency.phase ==
        residency::TiledGraphDependencyPhase::ReleaseComplete) {
      if (!release_valid_[bank] || release_history_[bank] != dependency.batch) {
        return false;
      }
      continue;
    }
    if (!terminal_valid_[bank][dependency.stage] ||
        terminal_history_[bank][dependency.stage].ordinal !=
            dependency.ordinal ||
        terminal_history_[bank][dependency.stage].batch != dependency.batch) {
      return false;
    }
  }
  return true;
}

bool Wavefront::select(WavefrontCoordinate &selected) const noexcept {
  const Cell *found = nullptr;
  for (const Cell &cell : cells_) {
    if (cell.state != CellState::Waiting || cell.failed || !cell.device_ready ||
        !predecessors_ready(cell) ||
        (found != nullptr && !before(cell.coordinate, found->coordinate))) {
      continue;
    }
    found = &cell;
  }
  if (found == nullptr) {
    selected = {};
    return false;
  }
  selected = found->coordinate;
  return true;
}

bool Wavefront::forecast(WavefrontCoordinate &selected,
                         std::uint32_t &resource) const noexcept {
  const Cell *found = nullptr;
  std::uint32_t found_resource = 0u;
  for (const Cell &cell : cells_) {
    if (cell.state != CellState::Waiting || cell.failed || cell.device_ready ||
        cell.required_mask == 0u ||
        cell.host_ready_mask == cell.required_mask ||
        !predecessors_ready(cell)) {
      continue;
    }
    std::uint32_t missing = 0u;
    bool has_missing = false;
    for (std::size_t index = 0u; index < cell.external_input_count; ++index) {
      const std::uint32_t bit = std::uint32_t{1u}
                                << static_cast<std::uint32_t>(index);
      if ((cell.host_ready_mask & bit) != 0u ||
          (cell.forecast_mask & bit) != 0u ||
          (has_missing && cell.external_inputs[index] >= missing)) {
        continue;
      }
      missing = cell.external_inputs[index];
      has_missing = true;
    }
    if (!has_missing ||
        (found != nullptr && !before(cell.coordinate, found->coordinate))) {
      continue;
    }
    found = &cell;
    found_resource = missing;
  }
  if (found == nullptr) {
    selected = {};
    resource = 0u;
    return false;
  }
  selected = found->coordinate;
  resource = found_resource;
  return true;
}

bool Wavefront::forecast_middle(const std::uint64_t batch,
                                WavefrontCoordinate &selected,
                                std::uint32_t &resource) const noexcept {
  const Cell *found = nullptr;
  std::uint32_t found_resource = 0u;
  for (const Cell &cell : cells_) {
    if (cell.coordinate.batch != batch || cell.coordinate.stage == 0u ||
        cell.coordinate.stage + 1u >= stage_count_ ||
        cell.state != CellState::Waiting || cell.failed || cell.device_ready ||
        cell.required_mask == 0u ||
        cell.host_ready_mask == cell.required_mask ||
        !predecessors_ready(cell)) {
      continue;
    }
    std::uint32_t missing = 0u;
    bool has_missing = false;
    for (std::size_t index = 0u; index < cell.external_input_count; ++index) {
      const std::uint32_t bit = std::uint32_t{1u}
                                << static_cast<std::uint32_t>(index);
      if ((cell.host_ready_mask & bit) != 0u ||
          (cell.forecast_mask & bit) != 0u ||
          (has_missing && cell.external_inputs[index] >= missing)) {
        continue;
      }
      missing = cell.external_inputs[index];
      has_missing = true;
    }
    if (!has_missing ||
        (found != nullptr && !before(cell.coordinate, found->coordinate))) {
      continue;
    }
    found = &cell;
    found_resource = missing;
  }
  if (found == nullptr) {
    selected = {};
    resource = 0u;
    return false;
  }
  selected = found->coordinate;
  resource = found_resource;
  return true;
}

bool Wavefront::pair_forecast(
    const std::uint64_t batch, std::array<WavefrontCoordinate, 2u> &coordinates,
    std::array<std::uint32_t, 2u> &resources) const noexcept {
  coordinates = {};
  resources = {};
  if (stage_count_ != 4u || invocation_ == nullptr ||
      batch >= invocation_->batch_count()) {
    return false;
  }

  const auto exact_missing_middle = [this](const Cell *const cell) noexcept {
    return cell != nullptr && cell->state == CellState::Waiting &&
           !cell->failed && !cell->device_ready && cell->required_mask == 1u &&
           cell->host_ready_mask == 0u && cell->forecast_mask == 0u &&
           cell->external_input_count == 1u && predecessors_ready(*cell) &&
           cell->coordinate.page_count != 0u;
  };
  const Cell *const first = find(batch, 1u);
  const Cell *const second = find(batch, 2u);
  if (!exact_missing_middle(first) || !exact_missing_middle(second) ||
      first->coordinate == second->coordinate ||
      first->external_inputs[0u] == second->external_inputs[0u]) {
    return false;
  }

  // No third current-batch external-input cell may be waiting behind the
  // pair. Stage zero is already terminal at this point and the terminal cell
  // is dependency-only for the admitted four-stage shape.
  for (const Cell &cell : cells_) {
    if (cell.state == CellState::Empty || cell.coordinate.batch != batch ||
        cell.coordinate.stage == 1u || cell.coordinate.stage == 2u) {
      continue;
    }
    if (cell.state == CellState::Waiting && cell.required_mask != 0u &&
        cell.host_ready_mask != cell.required_mask) {
      return false;
    }
  }

  coordinates[0u] = first->coordinate;
  coordinates[1u] = second->coordinate;
  resources[0u] = first->external_inputs[0u];
  resources[1u] = second->external_inputs[0u];
  return true;
}

bool Wavefront::reserve_forecast(const WavefrontCoordinate &coordinate,
                                 const std::uint32_t resource) noexcept {
  Cell *const cell = find(coordinate.batch, coordinate.stage);
  if (cell == nullptr || cell->state != CellState::Waiting || cell->failed ||
      cell->device_ready || cell->coordinate != coordinate ||
      !predecessors_ready(*cell)) {
    return false;
  }
  for (std::size_t index = 0u; index < cell->external_input_count; ++index) {
    if (cell->external_inputs[index] != resource) {
      continue;
    }
    const std::uint32_t bit = std::uint32_t{1u}
                              << static_cast<std::uint32_t>(index);
    if ((cell->host_ready_mask & bit) != 0u ||
        (cell->forecast_mask & bit) != 0u) {
      return false;
    }
    cell->forecast_mask |= bit;
    return true;
  }
  return false;
}

bool Wavefront::cancel_forecast(const WavefrontCoordinate &coordinate,
                                const std::uint32_t resource) noexcept {
  Cell *const cell = find(coordinate.batch, coordinate.stage);
  if (cell == nullptr || cell->state != CellState::Waiting ||
      cell->coordinate != coordinate) {
    return false;
  }
  for (std::size_t index = 0u; index < cell->external_input_count; ++index) {
    if (cell->external_inputs[index] != resource) {
      continue;
    }
    const std::uint32_t bit = std::uint32_t{1u}
                              << static_cast<std::uint32_t>(index);
    if ((cell->forecast_mask & bit) == 0u) {
      return false;
    }
    cell->forecast_mask &= ~bit;
    return true;
  }
  return false;
}

bool Wavefront::promote(const std::uint64_t batch,
                        WavefrontCoordinate &selected) const noexcept {
  const Cell *found = nullptr;
  for (const Cell &cell : cells_) {
    if (cell.coordinate.batch != batch || cell.coordinate.stage == 0u ||
        cell.coordinate.stage + 1u >= stage_count_ ||
        cell.state != CellState::Waiting || cell.failed || cell.device_ready ||
        cell.required_mask == 0u ||
        cell.host_ready_mask != cell.required_mask ||
        !predecessors_ready(cell) ||
        (found != nullptr && !before(cell.coordinate, found->coordinate))) {
      continue;
    }
    found = &cell;
  }
  if (found == nullptr) {
    selected = {};
    return false;
  }
  selected = found->coordinate;
  return true;
}

bool Wavefront::dispatch(const WavefrontCoordinate &coordinate) noexcept {
  Cell *const cell = find(coordinate.batch, coordinate.stage);
  if (cell == nullptr || cell->state != CellState::Waiting || cell->failed ||
      !cell->device_ready || !predecessors_ready(*cell) ||
      cell->coordinate != coordinate) {
    return false;
  }
  WavefrontCoordinate selected{};
  if (!select(selected) || selected != coordinate) {
    return false;
  }
  cell->state = CellState::Running;
  return true;
}

bool Wavefront::cancel_dispatch(
    const WavefrontCoordinate &coordinate) noexcept {
  Cell *const cell = find(coordinate.batch, coordinate.stage);
  if (cell == nullptr || cell->state != CellState::Running ||
      cell->coordinate != coordinate) {
    return false;
  }
  cell->state = CellState::Waiting;
  return true;
}

} // namespace rund::compute::detail::graph_reduce
