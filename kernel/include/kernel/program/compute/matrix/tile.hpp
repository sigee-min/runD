#pragma once

#include <kernel/core/checked.hpp>

namespace rund::kernel::matrix_tile {

inline constexpr u64 Side = 32u;
inline constexpr u64 RowsPerLane = 2u;
inline constexpr u64 ColsPerLane = 4u;
inline constexpr u64 Cells = Side * Side;
inline constexpr u64 Lanes = Cells / (RowsPerLane * ColsPerLane);
inline constexpr u64 LaneCols = Side / ColsPerLane;
static_assert(Cells == Lanes * RowsPerLane * ColsPerLane);

struct Tiles final {
  u64 count = 0u;
  bool ok = false;
};

[[nodiscard]] constexpr u64 Span(const u64 extent) noexcept {
  return extent / Side + static_cast<u64>(extent % Side != 0u);
}

[[nodiscard]] constexpr Tiles Count(const u64 rows, const u64 cols,
                                    const u64 batches) noexcept {
  const u64 tile_rows = Span(rows);
  const u64 tile_cols = Span(cols);
  u64 count = 0u;
  if (tile_rows == 0u || tile_cols == 0u || batches == 0u ||
      !checked::mul(tile_rows, tile_cols, batches, count)) {
    return {};
  }
  return Tiles{.count = count, .ok = true};
}

} // namespace rund::kernel::matrix_tile
