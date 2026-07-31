#pragma once

#include <string_view>

namespace rund::node::accel::detail::source::solve {

inline constexpr std::string_view Direct = R"GLSL(uint solve_direct_batch(uint64_t batch) {
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
  sync_solve();
  if (solve_code == 0u) { factorize_for_solve(n); }
  if (solve_code == 0u) {
    for (uint cell = lane; cell < n * rhs_cols; cell += 32u) {
      uint row = cell / rhs_cols;
      uint col = cell % rhs_cols;
      solve_x[cell] = rhs_values[ix(
          rhs_base + midx(row, col, p.rows, p.rhs_cols, p.storage_layout))];
    }
  }
  sync_solve();
  for (uint k = 0u; k < n; ++k) {
    if (lane == 0u && solve_code == 0u && solve_pivots[k] >= n) {
      solve_code = 1u;
    }
    sync_solve();
    if (solve_code == 0u && solve_pivots[k] != k) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int value = solve_x[k * rhs_cols + col];
        solve_x[k * rhs_cols + col] =
            solve_x[solve_pivots[k] * rhs_cols + col];
        solve_x[solve_pivots[k] * rhs_cols + col] = value;
      }
    }
    sync_solve();
  }
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
      solve_diag = solve_a[row * n + row];
      if (solve_diag == 0) { solve_code = 1u; }
    }
    sync_solve();
    if (solve_code == 0u) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int sum = solve_x[row * rhs_cols + col];
        for (uint k = row + 1u; k < n; ++k) {
          sum = sub_q31(
              sum, mul_q31(solve_a[row * n + k],
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

uint solve_direct_qr_batch(uint64_t batch) {
  uint lane = gl_LocalInvocationID.x;
  uint n = uint(p.rows);
  uint rhs_cols = uint(p.rhs_cols);
  uint64_t primary_base = batch * p.rows * p.rows;
  uint64_t rhs_base = batch * p.rows * p.rhs_cols;
  uint64_t out_base = batch * p.rows * p.rhs_cols;
  if (lane == 0u) {
    solve_code = (n > 16u || rhs_cols > 16u) ? 4u : 0u;
  }
  for (uint index = lane; index < 256u; index += 32u) {
    solve_a[index] = 0;
    solve_l[index] = 0;
  }
  sync_solve();
  for (uint col = 0u; col < n; ++col) {
    if (solve_code == 0u) {
      for (uint row = lane; row < n; row += 32u) {
        solve_multiplier[row] = primary_values[ix(
            primary_base + midx(row, col, p.rows, p.rows,
                                p.storage_layout))];
      }
    }
    sync_solve();
    for (uint previous = 0u; previous < col; ++previous) {
      if (lane == 0u && solve_code == 0u) {
        solve_diag = 0;
        for (uint row = 0u; row < n; ++row) {
          solve_diag = add_q31(
              solve_diag,
              mul_q31(solve_a[row * n + previous],
                      solve_multiplier[row]));
        }
        solve_l[previous * n + col] = solve_diag;
      }
      sync_solve();
      if (solve_code == 0u) {
        for (uint row = lane; row < n; row += 32u) {
          solve_multiplier[row] = sub_q31(
              solve_multiplier[row],
              mul_q31(solve_diag, solve_a[row * n + previous]));
        }
      }
      sync_solve();
    }
    if (lane == 0u && solve_code == 0u) {
      int squared = 0;
      for (uint row = 0u; row < n; ++row) {
        squared = add_q31(
            squared, mul_q31(solve_multiplier[row], solve_multiplier[row]));
      }
      solve_diag = sqrt_q31(squared);
      if (solve_diag == 0) {
        solve_code = 1u;
      } else {
        solve_l[col * n + col] = solve_diag;
      }
    }
    sync_solve();
    if (solve_code == 0u) {
      for (uint row = lane; row < n; row += 32u) {
        solve_a[row * n + col] =
            div_q31(solve_multiplier[row], solve_diag);
      }
    }
    sync_solve();
  }
  if (solve_code == 0u) {
    for (uint cell = lane; cell < n * rhs_cols; cell += 32u) {
      uint row = cell / rhs_cols;
      uint col = cell % rhs_cols;
      int sum = 0;
      for (uint k = 0u; k < n; ++k) {
        sum = add_q31(
            sum, mul_q31(
                     solve_a[k * n + row],
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
      solve_diag = solve_l[row * n + row];
      if (solve_diag == 0) { solve_code = 1u; }
    }
    sync_solve();
    if (solve_code == 0u) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int sum = solve_y[row * rhs_cols + col];
        for (uint k = row + 1u; k < n; ++k) {
          sum = sub_q31(sum,
                        mul_q31(solve_l[row * n + k],
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

uint solve_direct_cholesky_batch(uint64_t batch) {
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
    solve_l[index] = 0;
  }
  sync_solve();
  for (uint col = 0u; col < n; ++col) {
    if (lane == 0u && solve_code == 0u) {
      int sum = primary_values[ix(
          primary_base + midx(col, col, p.rows, p.rows,
                              p.storage_layout))];
      for (uint k = 0u; k < col; ++k) {
        int value = solve_l[col * n + k];
        sum = sub_q31(sum, mul_q31(value, value));
      }
      if (sum <= 0) {
        solve_code = 2u;
      } else {
        solve_diag = sqrt_q31(sum);
        solve_l[col * n + col] = solve_diag;
      }
    }
    sync_solve();
    if (solve_code == 0u) {
      for (uint row = col + 1u + lane; row < n; row += 32u) {
        int sum = primary_values[ix(
            primary_base + midx(row, col, p.rows, p.rows,
                                p.storage_layout))];
        for (uint k = 0u; k < col; ++k) {
          sum = sub_q31(
              sum, mul_q31(solve_l[row * n + k],
                           solve_l[col * n + k]));
        }
        solve_l[row * n + col] = div_q31(sum, solve_diag);
      }
    }
    sync_solve();
  }
  for (uint row = 0u; row < n; ++row) {
    if (lane == 0u && solve_code == 0u) {
      solve_diag = solve_l[row * n + row];
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
              sum, mul_q31(solve_l[row * n + k],
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
      solve_diag = solve_l[row * n + row];
      if (solve_diag == 0) { solve_code = 2u; }
    }
    sync_solve();
    if (solve_code == 0u) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int sum = solve_x[row * rhs_cols + col];
        for (uint k = row + 1u; k < n; ++k) {
          sum = sub_q31(
              sum, mul_q31(solve_l[k * n + row],
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

void main() {
  uint64_t batch = uint64_t(gl_WorkGroupID.x);
  if (batch >= p.batch_count) { return; }
  uint status = 4u;
  if (p.mode == 1u) {
    status = p.op == uint64_t(3) ? solve_direct_cholesky_batch(batch)
             : (p.op == uint64_t(2) ? solve_direct_qr_batch(batch)
                                     : solve_direct_batch(batch));
  } else if (p.op == uint64_t(1)) {
    status = solve_linear_batch(batch);
  } else if (p.op == uint64_t(3)) {
    status = solve_cholesky_batch(batch);
  } else if (p.op == uint64_t(2)) {
    status = solve_qr_batch(batch);
  }
  if (gl_LocalInvocationID.x == 0u) { status_values[ix(batch)] = status; }
}
)GLSL";

} // namespace rund::node::accel::detail::source::solve
