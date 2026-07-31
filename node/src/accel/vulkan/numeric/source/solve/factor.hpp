#pragma once

#include <string_view>

namespace rund::node::accel::detail::source::solve {

inline constexpr std::string_view Factor = R"GLSL(
layout(local_size_x = 32) in;
layout(set = 0, binding = 1, std430) readonly buffer Primary { int primary_values[]; };
layout(set = 0, binding = 2, std430) readonly buffer Aux { uint aux_values[]; };
layout(set = 0, binding = 3, std430) readonly buffer Rhs { int rhs_values[]; };
layout(set = 0, binding = 4, std430) buffer Output { int output_values[]; };
layout(set = 0, binding = 5, std430) buffer Status { uint status_values[]; };

shared int solve_a[256];
shared int solve_x[256];
shared int solve_y[256];
shared int solve_l[256];
shared int solve_multiplier[16];
shared int solve_diag;
shared uint solve_pivots[16];
shared uint solve_code;
shared uint solve_pivot;

void sync_solve() {
  memoryBarrierShared();
  memoryBarrierBuffer();
  barrier();
}

uint factorize_for_solve(uint n) {
  uint lane = gl_LocalInvocationID.x;
  for (uint k = 0u; k < n; ++k) {
    if (lane == 0u && solve_code == 0u) {
      uint pivot = k;
      if (p.aux == 2u) {
        uint best = 0u;
        for (uint row = k; row < n; ++row) {
          uint mag = mag_i32(
              solve_a[uint(midx(row, k, n, n, p.storage_layout))]);
          if (mag > best) {
            best = mag;
            pivot = row;
          }
        }
      }
      solve_pivot = pivot;
      solve_pivots[k] = pivot;
    }
    sync_solve();
    if (solve_code == 0u && solve_pivot != k) {
      for (uint c = lane; c < n; c += 32u) {
        uint li = uint(midx(k, c, n, n, p.storage_layout));
        uint ri = uint(midx(solve_pivot, c, n, n, p.storage_layout));
        int value = solve_a[li];
        solve_a[li] = solve_a[ri];
        solve_a[ri] = value;
      }
    }
    sync_solve();
    if (lane == 0u && solve_code == 0u) {
      solve_diag = solve_a[uint(midx(k, k, n, n, p.storage_layout))];
      if (solve_diag == 0) { solve_code = 1u; }
    }
    sync_solve();
    if (solve_code == 0u) {
      for (uint row = k + 1u + lane; row < n; row += 32u) {
        uint index = uint(midx(row, k, n, n, p.storage_layout));
        solve_a[index] = div_q31(solve_a[index], solve_diag);
      }
    }
    sync_solve();
    uint width = n - min(n, k + 1u);
    if (solve_code == 0u) {
      for (uint cell = lane; cell < width * width; cell += 32u) {
        uint row = k + 1u + cell / width;
        uint col = k + 1u + cell % width;
        uint index = uint(midx(row, col, n, n, p.storage_layout));
        uint multiplier = uint(midx(row, k, n, n, p.storage_layout));
        uint pivot = uint(midx(k, col, n, n, p.storage_layout));
        solve_a[index] = sub_q31(
            solve_a[index], mul_q31(solve_a[multiplier], solve_a[pivot]));
      }
    }
    sync_solve();
  }
  return solve_code;
}

uint solve_linear_batch(uint64_t batch) {
  uint lane = gl_LocalInvocationID.x;
  uint n = uint(p.rows);
  uint rhs_cols = uint(p.rhs_cols);
  uint64_t primary_base = batch * p.rows * p.rows;
  uint64_t rhs_base = batch * p.rows * p.rhs_cols;
  uint64_t out_base = batch * p.rows * p.rhs_cols;
  if (lane == 0u) {
    solve_code = (n > 16u || rhs_cols > 16u) ? 4u : 0u;
  }
  for (uint index = lane; index < n * n; index += 32u) {
    solve_a[index] = primary_values[ix(primary_base + index)];
  }
  if (p.mode == 2u) {
    for (uint index = lane; index < n; index += 32u) {
      solve_pivots[index] = aux_values[ix(batch * p.rows + index)];
    }
  }
  sync_solve();
  if (p.mode != 2u && solve_code == 0u) {
    factorize_for_solve(n);
  }
  if (solve_code == 0u) {
    for (uint cell = lane; cell < n * rhs_cols; cell += 32u) {
      uint row = cell / rhs_cols;
      uint col = cell % rhs_cols;
      uint source_row = p.mode == 2u ? solve_pivots[row] : row;
      solve_x[cell] = rhs_values[ix(
          rhs_base + midx(source_row, col, p.rows, p.rhs_cols,
                          p.storage_layout))];
    }
  }
  sync_solve();
  for (uint row = 0u; row < n; ++row) {
    if (solve_code == 0u) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int sum = solve_x[row * rhs_cols + col];
        for (uint k = 0u; k < row; ++k) {
          sum = sub_q31(
              sum, mul_q31(
                       solve_a[uint(midx(row, k, n, n,
                                         p.storage_layout))],
                       solve_x[k * rhs_cols + col]));
        }
        solve_x[row * rhs_cols + col] = sum;
      }
    }
    sync_solve();
  }
  for (int current = int(n) - 1; current >= 0; --current) {
    uint row = uint(current);
    if (lane == 0u && solve_code == 0u) {
      solve_diag = solve_a[uint(midx(row, row, n, n, p.storage_layout))];
      if (solve_diag == 0) { solve_code = 1u; }
    }
    sync_solve();
    if (solve_code == 0u) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int sum = solve_x[row * rhs_cols + col];
        for (uint k = row + 1u; k < n; ++k) {
          sum = sub_q31(
              sum, mul_q31(
                       solve_a[uint(midx(row, k, n, n,
                                         p.storage_layout))],
                       solve_x[k * rhs_cols + col]));
        }
        solve_x[row * rhs_cols + col] = div_q31(sum, solve_diag);
      }
    }
    sync_solve();
  }
  if (solve_code == 0u) {
    for (uint cell = lane; cell < n * rhs_cols; cell += 32u) {
      uint row = cell / rhs_cols;
      uint col = cell % rhs_cols;
      output_values[ix(out_base + midx(row, col, p.rows, p.rhs_cols,
                                       p.storage_layout))] = solve_x[cell];
    }
  }
  sync_solve();
  return solve_code;
}

uint solve_cholesky_batch(uint64_t batch) {
  uint lane = gl_LocalInvocationID.x;
  uint n = uint(p.rows);
  uint rhs_cols = uint(p.rhs_cols);
  uint64_t factor_base = batch * p.rows * p.rows;
  uint64_t rhs_base = batch * p.rows * p.rhs_cols;
  uint64_t out_base = batch * p.rows * p.rhs_cols;
  if (lane == 0u) {
    solve_code = (n > 16u || rhs_cols > 16u) ? 4u : 0u;
  }
  sync_solve();
  for (uint row = 0u; row < n; ++row) {
    if (lane == 0u && solve_code == 0u) {
      solve_diag = primary_values[ix(
          factor_base + midx(row, row, p.rows, p.rows, p.storage_layout))];
      if (solve_diag == 0) { solve_code = 2u; }
    }
    sync_solve();
    if (solve_code == 0u) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int sum = rhs_values[ix(
            rhs_base + midx(row, col, p.rows, p.rhs_cols,
                            p.storage_layout))];
        for (uint k = 0u; k < row; ++k) {
          sum = sub_q31(
              sum, mul_q31(
                       primary_values[ix(
                           factor_base + midx(row, k, p.rows, p.rows,
                                              p.storage_layout))],
                       solve_x[k * rhs_cols + col]));
        }
        solve_x[row * rhs_cols + col] = div_q31(sum, solve_diag);
      }
    }
    sync_solve();
  }
  for (int current = int(n) - 1; current >= 0; --current) {
    uint row = uint(current);
    if (lane == 0u && solve_code == 0u) {
      solve_diag = primary_values[ix(
          factor_base + midx(row, row, p.rows, p.rows, p.storage_layout))];
      if (solve_diag == 0) { solve_code = 2u; }
    }
    sync_solve();
    if (solve_code == 0u) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int sum = solve_x[row * rhs_cols + col];
        for (uint k = row + 1u; k < n; ++k) {
          sum = sub_q31(
              sum, mul_q31(
                       primary_values[ix(
                           factor_base + midx(k, row, p.rows, p.rows,
                                              p.storage_layout))],
                       solve_x[k * rhs_cols + col]));
        }
        solve_x[row * rhs_cols + col] = div_q31(sum, solve_diag);
      }
    }
    sync_solve();
  }
  if (solve_code == 0u) {
    for (uint cell = lane; cell < n * rhs_cols; cell += 32u) {
      uint row = cell / rhs_cols;
      uint col = cell % rhs_cols;
      output_values[ix(out_base + midx(row, col, p.rows, p.rhs_cols,
                                       p.storage_layout))] = solve_x[cell];
    }
  }
  sync_solve();
  return solve_code;
}

uint solve_qr_batch(uint64_t batch) {
  uint lane = gl_LocalInvocationID.x;
  uint n = uint(p.rows);
  uint rhs_cols = uint(p.rhs_cols);
  uint64_t factor_base = batch * p.rows * p.rows * uint64_t(2);
  uint64_t r_base = factor_base + p.rows * p.rows;
  uint64_t rhs_base = batch * p.rows * p.rhs_cols;
  uint64_t out_base = batch * p.rows * p.rhs_cols;
  if (lane == 0u) {
    solve_code = (n > 16u || rhs_cols > 16u) ? 4u : 0u;
  }
  sync_solve();
  if (solve_code == 0u) {
    for (uint cell = lane; cell < n * rhs_cols; cell += 32u) {
      uint row = cell / rhs_cols;
      uint col = cell % rhs_cols;
      int sum = 0;
      for (uint k = 0u; k < n; ++k) {
        sum = add_q31(
            sum, mul_q31(
                     primary_values[ix(
                         factor_base + midx(k, row, p.rows, p.rows,
                                            p.storage_layout))],
                     rhs_values[ix(
                         rhs_base + midx(k, col, p.rows, p.rhs_cols,
                                         p.storage_layout))]));
      }
      solve_y[cell] = sum;
    }
  }
  sync_solve();
  for (int current = int(n) - 1; current >= 0; --current) {
    uint row = uint(current);
    if (lane == 0u && solve_code == 0u) {
      solve_diag = primary_values[ix(
          r_base + midx(row, row, p.rows, p.rows, p.storage_layout))];
      if (solve_diag == 0) { solve_code = 1u; }
    }
    sync_solve();
    if (solve_code == 0u) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int sum = solve_y[row * rhs_cols + col];
        for (uint k = row + 1u; k < n; ++k) {
          sum = sub_q31(
              sum, mul_q31(
                       primary_values[ix(
                           r_base + midx(row, k, p.rows, p.rows,
                                         p.storage_layout))],
                       solve_x[k * rhs_cols + col]));
        }
        solve_x[row * rhs_cols + col] = div_q31(sum, solve_diag);
      }
    }
    sync_solve();
  }
  if (solve_code == 0u) {
    for (uint cell = lane; cell < n * rhs_cols; cell += 32u) {
      uint row = cell / rhs_cols;
      uint col = cell % rhs_cols;
      output_values[ix(out_base + midx(row, col, p.rows, p.rhs_cols,
                                       p.storage_layout))] = solve_x[cell];
    }
  }
  sync_solve();
  return solve_code;
}

)GLSL";

} // namespace rund::node::accel::detail::source::solve
