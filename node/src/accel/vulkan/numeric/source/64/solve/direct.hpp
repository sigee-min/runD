#pragma once

#include <string_view>

namespace rund::node::accel::detail::source::lane64::solve {

inline constexpr std::string_view Direct = R"GLSL(uint solve_direct_batch64(uint64_t batch) {
  uint lane = gl_LocalInvocationID.x;
  uint n = uint(p.rows);
  uint rhs_cols = uint(p.rhs_cols);
  uint64_t primary_base = batch * p.rows * p.rows;
  uint64_t rhs_base = batch * p.rows * p.rhs_cols;
  uint64_t out_base = batch * p.rows * p.rhs_cols;
  if (lane == 0u) {
    solve_code64 = (n > 16u || rhs_cols > 16u) ? 4u : 0u;
  }
  for (uint index = lane; index < n * n; index += 32u) {
    solve_a64[index] = primary_values[ix64(primary_base + index)];
  }
  sync_solve64();
  if (solve_code64 == 0u) { factorize_for_solve64(n); }
  if (solve_code64 == 0u) {
    for (uint cell = lane; cell < n * rhs_cols; cell += 32u) {
      uint row = cell / rhs_cols;
      uint col = cell % rhs_cols;
      solve_x64[cell] = rhs_values[ix64(
          rhs_base + midx64(row, col, p.rows, p.rhs_cols,
                            p.storage_layout))];
    }
  }
  sync_solve64();
  for (uint k = 0u; k < n; ++k) {
    if (lane == 0u && solve_code64 == 0u && solve_pivots64[k] >= n) {
      solve_code64 = 1u;
    }
    sync_solve64();
    if (solve_code64 == 0u && solve_pivots64[k] != k) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int64_t value = solve_x64[k * rhs_cols + col];
        solve_x64[k * rhs_cols + col] =
            solve_x64[solve_pivots64[k] * rhs_cols + col];
        solve_x64[solve_pivots64[k] * rhs_cols + col] = value;
      }
    }
    sync_solve64();
  }
  for (uint row = 0u; row < n; ++row) {
    if (solve_code64 == 0u) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int64_t sum = solve_x64[row * rhs_cols + col];
        for (uint k = 0u; k < row; ++k) {
          sum = sub_q63(
              sum, mul_q63(
                       solve_a64[uint(midx64(row, k, n, n,
                                             p.storage_layout))],
                       solve_x64[k * rhs_cols + col]));
        }
        solve_x64[row * rhs_cols + col] = sum;
      }
    }
    sync_solve64();
  }
  for (int current = int(n) - 1; current >= 0; --current) {
    uint row = uint(current);
    if (lane == 0u && solve_code64 == 0u) {
      solve_diag64 = solve_a64[row * n + row];
      if (solve_diag64 == int64_t(0)) { solve_code64 = 1u; }
    }
    sync_solve64();
    if (solve_code64 == 0u) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int64_t sum = solve_x64[row * rhs_cols + col];
        for (uint k = row + 1u; k < n; ++k) {
          sum = sub_q63(
              sum, mul_q63(solve_a64[row * n + k],
                           solve_x64[k * rhs_cols + col]));
        }
        solve_x64[row * rhs_cols + col] = div_q63(sum, solve_diag64);
      }
    }
    sync_solve64();
  }
  if (solve_code64 == 0u) {
    for (uint cell = lane; cell < n * rhs_cols; cell += 32u) {
      uint row = cell / rhs_cols;
      uint col = cell % rhs_cols;
      output_values[ix64(out_base + midx64(row, col, p.rows, p.rhs_cols,
                                           p.storage_layout))] =
          solve_x64[cell];
    }
  }
  sync_solve64();
  return solve_code64;
}

uint solve_direct_qr_batch64(uint64_t batch) {
  uint lane = gl_LocalInvocationID.x;
  uint n = uint(p.rows);
  uint rhs_cols = uint(p.rhs_cols);
  uint64_t primary_base = batch * p.rows * p.rows;
  uint64_t rhs_base = batch * p.rows * p.rhs_cols;
  uint64_t out_base = batch * p.rows * p.rhs_cols;
  if (lane == 0u) {
    solve_code64 = (n > 16u || rhs_cols > 16u) ? 4u : 0u;
  }
  for (uint index = lane; index < 256u; index += 32u) {
    solve_a64[index] = int64_t(0);
    solve_l64[index] = int64_t(0);
  }
  sync_solve64();
  for (uint col = 0u; col < n; ++col) {
    if (solve_code64 == 0u) {
      for (uint row = lane; row < n; row += 32u) {
        solve_multiplier64[row] = primary_values[ix64(
            primary_base + midx64(row, col, p.rows, p.rows,
                                  p.storage_layout))];
      }
    }
    sync_solve64();
    for (uint previous = 0u; previous < col; ++previous) {
      if (lane == 0u && solve_code64 == 0u) {
        solve_diag64 = int64_t(0);
        for (uint row = 0u; row < n; ++row) {
          solve_diag64 = add_q63(
              solve_diag64,
              mul_q63(solve_a64[row * n + previous],
                      solve_multiplier64[row]));
        }
        solve_l64[previous * n + col] = solve_diag64;
      }
      sync_solve64();
      if (solve_code64 == 0u) {
        for (uint row = lane; row < n; row += 32u) {
          solve_multiplier64[row] = sub_q63(
              solve_multiplier64[row],
              mul_q63(solve_diag64, solve_a64[row * n + previous]));
        }
      }
      sync_solve64();
    }
    if (lane == 0u && solve_code64 == 0u) {
      int64_t squared = int64_t(0);
      for (uint row = 0u; row < n; ++row) {
        squared = add_q63(
            squared,
            mul_q63(solve_multiplier64[row], solve_multiplier64[row]));
      }
      solve_diag64 = sqrt_q63(squared);
      if (solve_diag64 == int64_t(0)) {
        solve_code64 = 1u;
      } else {
        solve_l64[col * n + col] = solve_diag64;
      }
    }
    sync_solve64();
    if (solve_code64 == 0u) {
      for (uint row = lane; row < n; row += 32u) {
        solve_a64[row * n + col] =
            div_q63(solve_multiplier64[row], solve_diag64);
      }
    }
    sync_solve64();
  }
  if (solve_code64 == 0u) {
    for (uint cell = lane; cell < n * rhs_cols; cell += 32u) {
      uint row = cell / rhs_cols;
      uint col = cell % rhs_cols;
      int64_t sum = int64_t(0);
      for (uint k = 0u; k < n; ++k) {
        sum = add_q63(
            sum, mul_q63(
                     solve_a64[k * n + row],
                     rhs_values[ix64(
                         rhs_base + midx64(k, col, p.rows, p.rhs_cols,
                                           p.storage_layout))]));
      }
      solve_y64[cell] = sum;
    }
  }
  sync_solve64();
  for (int current = int(n) - 1; current >= 0; --current) {
    uint row = uint(current);
    if (lane == 0u && solve_code64 == 0u) {
      solve_diag64 = solve_l64[row * n + row];
      if (solve_diag64 == int64_t(0)) { solve_code64 = 1u; }
    }
    sync_solve64();
    if (solve_code64 == 0u) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int64_t sum = solve_y64[row * rhs_cols + col];
        for (uint k = row + 1u; k < n; ++k) {
          sum = sub_q63(sum,
                        mul_q63(solve_l64[row * n + k],
                                solve_x64[k * rhs_cols + col]));
        }
        solve_x64[row * rhs_cols + col] = div_q63(sum, solve_diag64);
      }
    }
    sync_solve64();
  }
  if (solve_code64 == 0u) {
    for (uint cell = lane; cell < n * rhs_cols; cell += 32u) {
      uint row = cell / rhs_cols;
      uint col = cell % rhs_cols;
      output_values[ix64(out_base + midx64(row, col, p.rows, p.rhs_cols,
                                           p.storage_layout))] =
          solve_x64[cell];
    }
  }
  sync_solve64();
  return solve_code64;
}

uint solve_direct_cholesky_batch64(uint64_t batch) {
  uint lane = gl_LocalInvocationID.x;
  uint n = uint(p.rows);
  uint rhs_cols = uint(p.rhs_cols);
  uint64_t primary_base = batch * p.rows * p.rows;
  uint64_t rhs_base = batch * p.rows * p.rhs_cols;
  uint64_t out_base = batch * p.rows * p.rhs_cols;
  if (lane == 0u) {
    solve_code64 = (n > 16u || rhs_cols > 16u) ? 4u : 0u;
  }
  for (uint index = lane; index < n * n; index += 32u) {
    solve_l64[index] = int64_t(0);
  }
  sync_solve64();
  for (uint col = 0u; col < n; ++col) {
    if (lane == 0u && solve_code64 == 0u) {
      int64_t sum = primary_values[ix64(
          primary_base + midx64(col, col, p.rows, p.rows,
                                p.storage_layout))];
      for (uint k = 0u; k < col; ++k) {
        int64_t value = solve_l64[col * n + k];
        sum = sub_q63(sum, mul_q63(value, value));
      }
      if (sum <= int64_t(0)) {
        solve_code64 = 2u;
      } else {
        solve_diag64 = sqrt_q63(sum);
        solve_l64[col * n + col] = solve_diag64;
      }
    }
    sync_solve64();
    if (solve_code64 == 0u) {
      for (uint row = col + 1u + lane; row < n; row += 32u) {
        int64_t sum = primary_values[ix64(
            primary_base + midx64(row, col, p.rows, p.rows,
                                  p.storage_layout))];
        for (uint k = 0u; k < col; ++k) {
          sum = sub_q63(
              sum, mul_q63(solve_l64[row * n + k],
                           solve_l64[col * n + k]));
        }
        solve_l64[row * n + col] = div_q63(sum, solve_diag64);
      }
    }
    sync_solve64();
  }
  for (uint row = 0u; row < n; ++row) {
    if (lane == 0u && solve_code64 == 0u) {
      solve_diag64 = solve_l64[row * n + row];
      if (solve_diag64 == int64_t(0)) { solve_code64 = 2u; }
    }
    sync_solve64();
    if (solve_code64 == 0u) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int64_t sum = rhs_values[ix64(
            rhs_base + midx64(row, col, p.rows, p.rhs_cols,
                              p.storage_layout))];
        for (uint k = 0u; k < row; ++k) {
          sum = sub_q63(
              sum, mul_q63(solve_l64[row * n + k],
                           solve_x64[k * rhs_cols + col]));
        }
        solve_x64[row * rhs_cols + col] = div_q63(sum, solve_diag64);
      }
    }
    sync_solve64();
  }
  for (int current = int(n) - 1; current >= 0; --current) {
    uint row = uint(current);
    if (lane == 0u && solve_code64 == 0u) {
      solve_diag64 = solve_l64[row * n + row];
      if (solve_diag64 == int64_t(0)) { solve_code64 = 2u; }
    }
    sync_solve64();
    if (solve_code64 == 0u) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int64_t sum = solve_x64[row * rhs_cols + col];
        for (uint k = row + 1u; k < n; ++k) {
          sum = sub_q63(
              sum, mul_q63(solve_l64[k * n + row],
                           solve_x64[k * rhs_cols + col]));
        }
        solve_x64[row * rhs_cols + col] = div_q63(sum, solve_diag64);
      }
    }
    sync_solve64();
  }
  if (solve_code64 == 0u) {
    for (uint cell = lane; cell < n * rhs_cols; cell += 32u) {
      uint row = cell / rhs_cols;
      uint col = cell % rhs_cols;
      output_values[ix64(out_base + midx64(row, col, p.rows, p.rhs_cols,
                                           p.storage_layout))] =
          solve_x64[cell];
    }
  }
  sync_solve64();
  return solve_code64;
}

void main() {
  uint64_t batch = uint64_t(gl_WorkGroupID.x);
  if (batch >= p.batch_count) { return; }
  uint status = 4u;
  if (p.mode == 1u) {
    status = p.op == uint64_t(3) ? solve_direct_cholesky_batch64(batch)
             : (p.op == uint64_t(2) ? solve_direct_qr_batch64(batch)
                                     : solve_direct_batch64(batch));
  } else if (p.op == uint64_t(1)) {
    status = solve_linear_batch64(batch);
  } else if (p.op == uint64_t(3)) {
    status = solve_cholesky_batch64(batch);
  } else if (p.op == uint64_t(2)) {
    status = solve_qr_batch64(batch);
  }
  if (gl_LocalInvocationID.x == 0u) { status_values[ix64(batch)] = status; }
}
)GLSL";

} // namespace rund::node::accel::detail::source::lane64::solve
