#include "../wavefront.hpp"

namespace rund::compute::detail::graph_reduce {

bool Wavefront::terminal(const WavefrontCoordinate &coordinate) noexcept {
  Cell *const cell = find(coordinate.batch, coordinate.stage);
  if (cell == nullptr || cell->state != CellState::Running ||
      cell->coordinate != coordinate) {
    return false;
  }
  cell->state = CellState::Terminal;
  const std::size_t bank =
      static_cast<std::size_t>(coordinate.batch % WavefrontBankCount);
  terminal_history_[bank][coordinate.stage] = coordinate;
  terminal_valid_[bank][coordinate.stage] = true;
  return true;
}

bool Wavefront::terminal(const std::uint64_t batch,
                         const std::uint32_t stage) noexcept {
  Cell *const cell = find(batch, stage);
  return cell != nullptr && terminal(cell->coordinate);
}

bool Wavefront::release(const std::uint64_t batch) noexcept {
  const std::size_t bank = static_cast<std::size_t>(batch % WavefrontBankCount);
  for (std::size_t stage = 0u; stage < stage_count_; ++stage) {
    const Cell &cell = cells_[bank * WavefrontStageCapacity + stage];
    if (cell.state != CellState::Terminal || cell.coordinate.batch != batch) {
      return false;
    }
  }
  for (std::size_t stage = 0u; stage < stage_count_; ++stage) {
    cells_[bank * WavefrontStageCapacity + stage] = {};
  }
  release_history_[bank] = batch;
  release_valid_[bank] = true;
  return true;
}

void Wavefront::abort() noexcept {
  cells_ = {};
  terminal_history_ = {};
  terminal_valid_ = {};
  release_history_ = {};
  release_valid_ = {};
}

} // namespace rund::compute::detail::graph_reduce
