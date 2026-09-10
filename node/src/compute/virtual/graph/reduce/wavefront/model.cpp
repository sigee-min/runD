#include "../wavefront.hpp"

namespace rund::compute::detail::graph_reduce {

Wavefront::Cell *Wavefront::find(const std::uint64_t batch,
                                 const std::uint32_t stage) noexcept {
  if (stage >= stage_count_) {
    return nullptr;
  }
  Cell &cell = cells_[static_cast<std::size_t>(batch % WavefrontBankCount) *
                          WavefrontStageCapacity +
                      stage];
  return cell.state != CellState::Empty && cell.coordinate.batch == batch &&
                 cell.coordinate.stage == stage
             ? &cell
             : nullptr;
}

const Wavefront::Cell *
Wavefront::find(const std::uint64_t batch,
                const std::uint32_t stage) const noexcept {
  return const_cast<Wavefront *>(this)->find(batch, stage);
}

bool Wavefront::before(const WavefrontCoordinate &left,
                       const WavefrontCoordinate &right) noexcept {
  if (left.stage != right.stage) {
    return left.stage < right.stage;
  }
  if (left.ordinal != right.ordinal) {
    return left.ordinal < right.ordinal;
  }
  return left.resource < right.resource;
}

} // namespace rund::compute::detail::graph_reduce
