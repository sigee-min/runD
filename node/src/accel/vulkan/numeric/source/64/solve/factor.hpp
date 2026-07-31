#pragma once

#include <string_view>

namespace rund::node::accel::detail::source::lane64::solve {

inline constexpr std::string_view Factor = R"GLSL(
layout(local_size_x = 32) in;
layout(set = 0, binding = 1, std430) readonly buffer Primary { int64_t primary_values[]; };
layout(set = 0, binding = 2, std430) readonly buffer Aux { uint aux_values[]; };
layout(set = 0, binding = 3, std430) readonly buffer Rhs { int64_t rhs_values[]; };
layout(set = 0, binding = 4, std430) buffer Output { int64_t output_values[]; };
layout(set = 0, binding = 5, std430) buffer Status { uint status_values[]; };

shared int64_t solve_a64[256];
shared int64_t solve_x64[256];
shared int64_t solve_y64[256];
shared int64_t solve_l64[256];
shared int64_t solve_multiplier64[16];
shared int64_t solve_diag64;
shared uint solve_pivots64[16];
shared uint solve_code64;
shared uint solve_pivot64;

void sync_solve64() {
  memoryBarrierShared();
  memoryBarrierBuffer();
  barrier();
}

uint factorize_for_solve64(uint n) {
  uint lane = gl_LocalInvocationID.x;
  for (uint k = 0u; k < n; ++k) {
    if (lane == 0u && solve_code64 == 0u) {
      uint pivot = k;
      if (p.aux == 2u) {
        uint64_t best = uint64_t(0);
        for (uint row = k; row < n; ++row) {
          uint64_t magnitude = RundAbsMagnitude64(as_u64(
              solve_a64[uint(midx64(row, k, n, n, p.storage_layout))]));
          if (magnitude > best) {
            best = magnitude;
            pivot = row;
          }
        }
      }
      solve_pivot64 = pivot;
      solve_pivots64[k] = pivot;
    }
    sync_solve64();
    if (solve_code64 == 0u && solve_pivot64 != k) {
      for (uint c = lane; c < n; c += 32u) {
        uint li = uint(midx64(k, c, n, n, p.storage_layout));
        uint ri = uint(midx64(solve_pivot64, c, n, n,
                              p.storage_layout));
        int64_t value = solve_a64[li];
        solve_a64[li] = solve_a64[ri];
        solve_a64[ri] = value;
      }
    }
    sync_solve64();
    if (lane == 0u && solve_code64 == 0u) {
      solve_diag64 =
          solve_a64[uint(midx64(k, k, n, n, p.storage_layout))];
      if (solve_diag64 == int64_t(0)) { solve_code64 = 1u; }
    }
    sync_solve64();
    if (solve_code64 == 0u) {
      for (uint row = k + 1u + lane; row < n; row += 32u) {
        uint index = uint(midx64(row, k, n, n, p.storage_layout));
        solve_a64[index] = div_q63(solve_a64[index], solve_diag64);
      }
    }
    sync_solve64();
    uint width = n - min(n, k + 1u);
    if (solve_code64 == 0u) {
      for (uint cell = lane; cell < width * width; cell += 32u) {
        uint row = k + 1u + cell / width;
        uint col = k + 1u + cell % width;
        uint index = uint(midx64(row, col, n, n, p.storage_layout));
        uint multiplier = uint(midx64(row, k, n, n, p.storage_layout));
        uint pivot = uint(midx64(k, col, n, n, p.storage_layout));
        solve_a64[index] = sub_q63(
            solve_a64[index],
            mul_q63(solve_a64[multiplier], solve_a64[pivot]));
      }
    }
    sync_solve64();
  }
  return solve_code64;
}

uint solve_linear_batch64(uint64_t batch) {
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
  if (p.mode == 2u) {
    for (uint index = lane; index < n; index += 32u) {
      solve_pivots64[index] = aux_values[ix64(batch * p.rows + index)];
    }
  }
  sync_solve64();
  if (p.mode != 2u && solve_code64 == 0u) {
    factorize_for_solve64(n);
  }
  if (solve_code64 == 0u) {
    for (uint cell = lane; cell < n * rhs_cols; cell += 32u) {
      uint row = cell / rhs_cols;
      uint col = cell % rhs_cols;
      uint source_row = p.mode == 2u ? solve_pivots64[row] : row;
      solve_x64[cell] = rhs_values[ix64(
          rhs_base + midx64(source_row, col, p.rows, p.rhs_cols,
                            p.storage_layout))];
    }
  }
  sync_solve64();
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
      solve_diag64 =
          solve_a64[uint(midx64(row, row, n, n, p.storage_layout))];
      if (solve_diag64 == int64_t(0)) { solve_code64 = 1u; }
    }
    sync_solve64();
    if (solve_code64 == 0u) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int64_t sum = solve_x64[row * rhs_cols + col];
        for (uint k = row + 1u; k < n; ++k) {
          sum = sub_q63(
              sum, mul_q63(
                       solve_a64[uint(midx64(row, k, n, n,
                                             p.storage_layout))],
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

uint solve_cholesky_batch64(uint64_t batch) {
  uint lane = gl_LocalInvocationID.x;
  uint n = uint(p.rows);
  uint rhs_cols = uint(p.rhs_cols);
  uint64_t factor_base = batch * p.rows * p.rows;
  uint64_t rhs_base = batch * p.rows * p.rhs_cols;
  uint64_t out_base = batch * p.rows * p.rhs_cols;
  if (lane == 0u) {
    solve_code64 = (n > 16u || rhs_cols > 16u) ? 4u : 0u;
  }
  sync_solve64();
  for (uint row = 0u; row < n; ++row) {
    if (lane == 0u && solve_code64 == 0u) {
      solve_diag64 = primary_values[ix64(
          factor_base + midx64(row, row, p.rows, p.rows,
                               p.storage_layout))];
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
              sum, mul_q63(
                       primary_values[ix64(
                           factor_base + midx64(row, k, p.rows, p.rows,
                                                p.storage_layout))],
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
      solve_diag64 = primary_values[ix64(
          factor_base + midx64(row, row, p.rows, p.rows,
                               p.storage_layout))];
      if (solve_diag64 == int64_t(0)) { solve_code64 = 2u; }
    }
    sync_solve64();
    if (solve_code64 == 0u) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int64_t sum = solve_x64[row * rhs_cols + col];
        for (uint k = row + 1u; k < n; ++k) {
          sum = sub_q63(
              sum, mul_q63(
                       primary_values[ix64(
                           factor_base + midx64(k, row, p.rows, p.rows,
                                                p.storage_layout))],
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

uint solve_qr_batch64(uint64_t batch) {
  uint lane = gl_LocalInvocationID.x;
  uint n = uint(p.rows);
  uint rhs_cols = uint(p.rhs_cols);
  uint64_t factor_base = batch * p.rows * p.rows * uint64_t(2);
  uint64_t r_base = factor_base + p.rows * p.rows;
  uint64_t rhs_base = batch * p.rows * p.rhs_cols;
  uint64_t out_base = batch * p.rows * p.rhs_cols;
  if (lane == 0u) {
    solve_code64 = (n > 16u || rhs_cols > 16u) ? 4u : 0u;
  }
  sync_solve64();
  if (solve_code64 == 0u) {
    for (uint cell = lane; cell < n * rhs_cols; cell += 32u) {
      uint row = cell / rhs_cols;
      uint col = cell % rhs_cols;
      int64_t sum = int64_t(0);
      for (uint k = 0u; k < n; ++k) {
        sum = add_q63(
            sum, mul_q63(
                     primary_values[ix64(
                         factor_base + midx64(k, row, p.rows, p.rows,
                                              p.storage_layout))],
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
      solve_diag64 = primary_values[ix64(
          r_base + midx64(row, row, p.rows, p.rows, p.storage_layout))];
      if (solve_diag64 == int64_t(0)) { solve_code64 = 1u; }
    }
    sync_solve64();
    if (solve_code64 == 0u) {
      for (uint col = lane; col < rhs_cols; col += 32u) {
        int64_t sum = solve_y64[row * rhs_cols + col];
        for (uint k = row + 1u; k < n; ++k) {
          sum = sub_q63(
              sum, mul_q63(
                       primary_values[ix64(
                           r_base + midx64(row, k, p.rows, p.rows,
                                           p.storage_layout))],
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

)GLSL";

} // namespace rund::node::accel::detail::source::lane64::solve
