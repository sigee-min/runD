#pragma once

#include <kernel/program/compute/matrix/tile.hpp>
#include <string>
#include <string_view>

namespace rund::node::accel::detail {

[[nodiscard]] inline std::string
MatrixProgramSource(std::string base, const std::string_view dialect) {
  const auto define = [&base](const std::string_view name,
                              const kernel::u64 value) {
    base += "#define ";
    base += name;
    base += ' ';
    base += std::to_string(value);
    base += "u\n";
  };
  define("RUND_MATRIX_TILE_SIDE", kernel::matrix_tile::Side);
  define("RUND_MATRIX_TILE_CELLS", kernel::matrix_tile::Cells);
  define("RUND_MATRIX_TILE_LANES", kernel::matrix_tile::Lanes);
  define("RUND_MATRIX_TILE_ROWS", kernel::matrix_tile::RowsPerLane);
  define("RUND_MATRIX_TILE_COLS", kernel::matrix_tile::ColsPerLane);
  define("RUND_MATRIX_LANE_COLS", kernel::matrix_tile::LaneCols);
  base.append(dialect);
  base.append(R"GLSL(
layout(local_size_x = RUND_MATRIX_TILE_LANES) in;
layout(set = 0, binding = 1, std430) readonly buffer Left {
  RUND_MATRIX_SCALAR left_values[];
};
layout(set = 0, binding = 2, std430) readonly buffer Right {
  RUND_MATRIX_SCALAR right_values[];
};
layout(set = 0, binding = 3, std430) buffer Output {
  RUND_MATRIX_SCALAR output_values[];
};

shared RUND_MATRIX_SCALAR left_tile[RUND_MATRIX_TILE_CELLS];
shared RUND_MATRIX_SCALAR right_tile[RUND_MATRIX_TILE_CELLS];

void sync_matrix() {
  memoryBarrierShared();
  barrier();
}

void main() {
  uint lane = gl_LocalInvocationID.x;
  uint64_t tile_cols =
      (p.cols + uint64_t(RUND_MATRIX_TILE_SIDE - 1u)) /
      uint64_t(RUND_MATRIX_TILE_SIDE);
  uint64_t tile_rows =
      (p.rows + uint64_t(RUND_MATRIX_TILE_SIDE - 1u)) /
      uint64_t(RUND_MATRIX_TILE_SIDE);
  uint64_t tiles_per_batch = tile_rows * tile_cols;
  uint64_t tile = uint64_t(gl_WorkGroupID.x);
  uint64_t batch = tile / tiles_per_batch;
  uint64_t cell = tile % tiles_per_batch;
  uint64_t row_tile =
      (cell / tile_cols) * uint64_t(RUND_MATRIX_TILE_SIDE);
  uint64_t col_tile =
      (cell % tile_cols) * uint64_t(RUND_MATRIX_TILE_SIDE);

  if (p.op == uint64_t(2)) {
    for (uint item = lane; item < RUND_MATRIX_TILE_CELLS;
         item += RUND_MATRIX_TILE_LANES) {
      uint64_t row = row_tile + uint64_t(item / RUND_MATRIX_TILE_SIDE);
      uint64_t col = col_tile + uint64_t(item % RUND_MATRIX_TILE_SIDE);
      if (row < p.rows && col < p.cols) {
        uint source = RUND_MATRIX_INDEX(
            batch * p.rows * p.cols +
            RUND_MATRIX_MIDX(row, col, p.rows, p.cols, p.storage_layout));
        uint target = RUND_MATRIX_INDEX(
            batch * p.rows * p.cols +
            RUND_MATRIX_MIDX(col, row, p.cols, p.rows, p.storage_layout));
        output_values[target] = left_values[source];
      }
    }
    return;
  }

  uint64_t local_row =
      uint64_t(lane / RUND_MATRIX_LANE_COLS) * RUND_MATRIX_TILE_ROWS;
  uint64_t local_col =
      uint64_t(lane % RUND_MATRIX_LANE_COLS) * RUND_MATRIX_TILE_COLS;
  uint64_t row = row_tile + local_row;
  uint64_t col = col_tile + local_col;
  uint64_t left_base = batch * p.rows * p.inner;
  uint64_t right_base = batch * p.inner * p.cols;
  RUND_MATRIX_SCALAR sum00 = RUND_MATRIX_ZERO;
  RUND_MATRIX_SCALAR sum01 = RUND_MATRIX_ZERO;
  RUND_MATRIX_SCALAR sum02 = RUND_MATRIX_ZERO;
  RUND_MATRIX_SCALAR sum03 = RUND_MATRIX_ZERO;
  RUND_MATRIX_SCALAR sum10 = RUND_MATRIX_ZERO;
  RUND_MATRIX_SCALAR sum11 = RUND_MATRIX_ZERO;
  RUND_MATRIX_SCALAR sum12 = RUND_MATRIX_ZERO;
  RUND_MATRIX_SCALAR sum13 = RUND_MATRIX_ZERO;
  for (uint64_t base = uint64_t(0); base < p.inner;
       base += uint64_t(RUND_MATRIX_TILE_SIDE)) {
    for (uint item = lane; item < RUND_MATRIX_TILE_CELLS;
         item += RUND_MATRIX_TILE_LANES) {
      uint64_t item_row = uint64_t(item / RUND_MATRIX_TILE_SIDE);
      uint64_t item_col = uint64_t(item % RUND_MATRIX_TILE_SIDE);
      uint64_t left_row = row_tile + item_row;
      uint64_t left_k = base + item_col;
      uint64_t right_k = base + item_row;
      uint64_t right_col = col_tile + item_col;
      left_tile[item] = left_row < p.rows && left_k < p.inner
                            ? left_values[RUND_MATRIX_INDEX(
                                  left_base + RUND_MATRIX_MIDX(
                                                  left_row, left_k, p.rows,
                                                  p.inner, p.storage_layout))]
                            : RUND_MATRIX_ZERO;
      right_tile[item] = right_k < p.inner && right_col < p.cols
                             ? right_values[RUND_MATRIX_INDEX(
                                   right_base + RUND_MATRIX_MIDX(
                                                    right_k, right_col,
                                                    p.inner, p.cols,
                                                    p.storage_layout))]
                             : RUND_MATRIX_ZERO;
    }
    sync_matrix();
    uint64_t remaining = p.inner - base;
    uint64_t width = min(uint64_t(RUND_MATRIX_TILE_SIDE), remaining);
    for (uint64_t offset = uint64_t(0); offset < width; ++offset) {
      RUND_MATRIX_SCALAR left0 = left_tile[uint(
          local_row * uint64_t(RUND_MATRIX_TILE_SIDE) + offset)];
      RUND_MATRIX_SCALAR left1 = left_tile[uint(
          (local_row + uint64_t(1)) * uint64_t(RUND_MATRIX_TILE_SIDE) +
          offset)];
      uint right_index = uint(
          offset * uint64_t(RUND_MATRIX_TILE_SIDE) + local_col);
      RUND_MATRIX_SCALAR right0 = right_tile[right_index];
      RUND_MATRIX_SCALAR right1 = right_tile[right_index + 1u];
      RUND_MATRIX_SCALAR right2 = right_tile[right_index + 2u];
      RUND_MATRIX_SCALAR right3 = right_tile[right_index + 3u];
      sum00 = RUND_MATRIX_ADD(sum00, RUND_MATRIX_MUL(left0, right0));
      sum01 = RUND_MATRIX_ADD(sum01, RUND_MATRIX_MUL(left0, right1));
      sum02 = RUND_MATRIX_ADD(sum02, RUND_MATRIX_MUL(left0, right2));
      sum03 = RUND_MATRIX_ADD(sum03, RUND_MATRIX_MUL(left0, right3));
      sum10 = RUND_MATRIX_ADD(sum10, RUND_MATRIX_MUL(left1, right0));
      sum11 = RUND_MATRIX_ADD(sum11, RUND_MATRIX_MUL(left1, right1));
      sum12 = RUND_MATRIX_ADD(sum12, RUND_MATRIX_MUL(left1, right2));
      sum13 = RUND_MATRIX_ADD(sum13, RUND_MATRIX_MUL(left1, right3));
    }
    sync_matrix();
  }
  uint64_t output_base = batch * p.rows * p.cols;
#define RUND_STORE_MATRIX(row_offset, col_offset, value)                    \
  if (row + uint64_t(row_offset) < p.rows &&                               \
      col + uint64_t(col_offset) < p.cols) {                               \
    uint target = RUND_MATRIX_INDEX(                                       \
        output_base + RUND_MATRIX_MIDX(                                    \
                          row + uint64_t(row_offset),                       \
                          col + uint64_t(col_offset), p.rows, p.cols,       \
                          p.storage_layout));                              \
    output_values[target] = value;                                         \
  }
  RUND_STORE_MATRIX(0, 0, sum00)
  RUND_STORE_MATRIX(0, 1, sum01)
  RUND_STORE_MATRIX(0, 2, sum02)
  RUND_STORE_MATRIX(0, 3, sum03)
  RUND_STORE_MATRIX(1, 0, sum10)
  RUND_STORE_MATRIX(1, 1, sum11)
  RUND_STORE_MATRIX(1, 2, sum12)
  RUND_STORE_MATRIX(1, 3, sum13)
#undef RUND_STORE_MATRIX
}
)GLSL");
  return base;
}

} // namespace rund::node::accel::detail
