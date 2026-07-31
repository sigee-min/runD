#pragma once

#include <string_view>

namespace rund::node::accel::detail::source::program {

inline constexpr std::string_view Matrix = R"MSL(

kernel void RUND_KERNEL(rund_numeric_matrix_)(
    device const RUND_SCALAR* left [[buffer(0)]],
    device const RUND_SCALAR* right [[buffer(1)]],
    device RUND_SCALAR* output [[buffer(2)]],
    constant NumericParams& p [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  ulong tile_cols = (p.cols + RUND_MATRIX_TILE_SIDE - 1ul) /
                    RUND_MATRIX_TILE_SIDE;
  ulong tile_rows = (p.rows + RUND_MATRIX_TILE_SIDE - 1ul) /
                    RUND_MATRIX_TILE_SIDE;
  ulong tiles_per_batch = tile_rows * tile_cols;
  ulong tile = ulong(group.x);
  ulong b = tile / tiles_per_batch;
  ulong cell = tile % tiles_per_batch;
  ulong row_tile = (cell / tile_cols) * RUND_MATRIX_TILE_SIDE;
  ulong col_tile = (cell % tile_cols) * RUND_MATRIX_TILE_SIDE;
  if (p.op == 2ul) {
    for (uint item = lane; item < RUND_MATRIX_TILE_CELLS;
         item += RUND_MATRIX_TILE_LANES) {
      ulong row = row_tile + ulong(item / RUND_MATRIX_TILE_SIDE);
      ulong col = col_tile + ulong(item % RUND_MATRIX_TILE_SIDE);
      if (row < p.rows && col < p.cols) {
        output[b * p.rows * p.cols +
               RUND_INDEX(col, row, p.cols, p.rows, p.layout)] =
            left[b * p.rows * p.cols +
                 RUND_INDEX(row, col, p.rows, p.cols, p.layout)];
      }
    }
    return;
  }
  ulong local_row = ulong(lane / RUND_MATRIX_LANE_COLS) *
                    RUND_MATRIX_TILE_ROWS;
  ulong local_col = ulong(lane % RUND_MATRIX_LANE_COLS) *
                    RUND_MATRIX_TILE_COLS;
  ulong row = row_tile + local_row;
  ulong col = col_tile + local_col;
  device const RUND_SCALAR* a = left + b * p.rows * p.inner;
  device const RUND_SCALAR* r = right + b * p.inner * p.cols;
  threadgroup RUND_SCALAR left_tile[RUND_MATRIX_TILE_CELLS];
  threadgroup RUND_SCALAR right_tile[RUND_MATRIX_TILE_CELLS];
  RUND_SCALAR sum00 = RUND_ZERO;
  RUND_SCALAR sum01 = RUND_ZERO;
  RUND_SCALAR sum02 = RUND_ZERO;
  RUND_SCALAR sum03 = RUND_ZERO;
  RUND_SCALAR sum10 = RUND_ZERO;
  RUND_SCALAR sum11 = RUND_ZERO;
  RUND_SCALAR sum12 = RUND_ZERO;
  RUND_SCALAR sum13 = RUND_ZERO;
  for (ulong base = 0ul; base < p.inner;
       base += RUND_MATRIX_TILE_SIDE) {
    for (uint item = lane; item < RUND_MATRIX_TILE_CELLS;
         item += RUND_MATRIX_TILE_LANES) {
      ulong item_row = ulong(item / RUND_MATRIX_TILE_SIDE);
      ulong item_col = ulong(item % RUND_MATRIX_TILE_SIDE);
      ulong left_row = row_tile + item_row;
      ulong left_k = base + item_col;
      ulong right_k = base + item_row;
      ulong right_col = col_tile + item_col;
      left_tile[item] = left_row < p.rows && left_k < p.inner
                            ? a[RUND_INDEX(left_row, left_k, p.rows, p.inner,
                                           p.layout)]
                            : RUND_ZERO;
      right_tile[item] = right_k < p.inner && right_col < p.cols
                             ? r[RUND_INDEX(right_k, right_col, p.inner,
                                            p.cols, p.layout)]
                             : RUND_ZERO;
    }
    rund_numeric_sync();
    ulong width = min(ulong(RUND_MATRIX_TILE_SIDE), p.inner - base);
    // Eight outputs share each staged value while every output retains the
    // exact ascending k fold required by fixed rounding and saturation.
    for (ulong offset = 0ul; offset < width; ++offset) {
      RUND_SCALAR left0 =
          left_tile[local_row * RUND_MATRIX_TILE_SIDE + offset];
      RUND_SCALAR left1 =
          left_tile[(local_row + 1ul) * RUND_MATRIX_TILE_SIDE + offset];
      RUND_SCALAR right0 =
          right_tile[offset * RUND_MATRIX_TILE_SIDE + local_col];
      RUND_SCALAR right1 =
          right_tile[offset * RUND_MATRIX_TILE_SIDE + local_col + 1ul];
      RUND_SCALAR right2 =
          right_tile[offset * RUND_MATRIX_TILE_SIDE + local_col + 2ul];
      RUND_SCALAR right3 =
          right_tile[offset * RUND_MATRIX_TILE_SIDE + local_col + 3ul];
      sum00 = RUND_MATRIX_ADD(sum00, RUND_MATRIX_MUL(left0, right0, p), p);
      sum01 = RUND_MATRIX_ADD(sum01, RUND_MATRIX_MUL(left0, right1, p), p);
      sum02 = RUND_MATRIX_ADD(sum02, RUND_MATRIX_MUL(left0, right2, p), p);
      sum03 = RUND_MATRIX_ADD(sum03, RUND_MATRIX_MUL(left0, right3, p), p);
      sum10 = RUND_MATRIX_ADD(sum10, RUND_MATRIX_MUL(left1, right0, p), p);
      sum11 = RUND_MATRIX_ADD(sum11, RUND_MATRIX_MUL(left1, right1, p), p);
      sum12 = RUND_MATRIX_ADD(sum12, RUND_MATRIX_MUL(left1, right2, p), p);
      sum13 = RUND_MATRIX_ADD(sum13, RUND_MATRIX_MUL(left1, right3, p), p);
    }
    rund_numeric_sync();
  }
  ulong output_base = b * p.rows * p.cols;
#define RUND_STORE_MATRIX(row_offset, col_offset, value)                    \
  if (row + row_offset < p.rows && col + col_offset < p.cols) {            \
    output[output_base +                                                   \
           RUND_INDEX(row + row_offset, col + col_offset, p.rows, p.cols,  \
                      p.layout)] = value;                                  \
  }
  RUND_STORE_MATRIX(0ul, 0ul, sum00)
  RUND_STORE_MATRIX(0ul, 1ul, sum01)
  RUND_STORE_MATRIX(0ul, 2ul, sum02)
  RUND_STORE_MATRIX(0ul, 3ul, sum03)
  RUND_STORE_MATRIX(1ul, 0ul, sum10)
  RUND_STORE_MATRIX(1ul, 1ul, sum11)
  RUND_STORE_MATRIX(1ul, 2ul, sum12)
  RUND_STORE_MATRIX(1ul, 3ul, sum13)
#undef RUND_STORE_MATRIX
}

)MSL";

} // namespace rund::node::accel::detail::source::program
