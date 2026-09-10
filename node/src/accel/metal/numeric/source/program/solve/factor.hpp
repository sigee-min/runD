#pragma once

#include <string_view>

namespace rund::node::accel::detail::source::program::solve {

inline constexpr std::string_view Factor =
    R"MSL(inline void rund_solve_factor_lu(
    device const RUND_SCALAR* factor, device const uint* pivots,
    device const RUND_SCALAR* rhs, device RUND_SCALAR* output,
    constant NumericParams& p, ulong batch, uint lane, uint lanes,
    threadgroup uint* control) {
  ulong n = p.rows;
  ulong rc = p.rhs_cols;
  device RUND_SCALAR* x = output + batch * n * rc;
  device const RUND_SCALAR* b = rhs + batch * n * rc;
  for (ulong cell = ulong(lane); cell < n * rc; cell += ulong(lanes)) {
    ulong row = cell / rc;
    ulong col = cell % rc;
    x[RUND_INDEX(row, col, n, rc, p.layout)] =
        b[RUND_INDEX(row, col, n, rc, p.layout)];
  }
  if (lane == 0u) { control[0] = 0u; }
  rund_numeric_sync();
  for (ulong k = 0ul; k < n; ++k) {
    if (lane == 0u) {
      control[1] = pivots[batch * n + k];
      if (ulong(control[1]) >= n) { control[0] = 1u; }
    }
    rund_numeric_sync();
    if (control[0] != 0u) { break; }
    ulong pivot = ulong(control[1]);
    if (pivot != k) {
      for (ulong col = ulong(lane); col < rc; col += ulong(lanes)) {
        ulong lhs = RUND_INDEX(k, col, n, rc, p.layout);
        ulong other = RUND_INDEX(pivot, col, n, rc, p.layout);
        RUND_SCALAR value = x[lhs];
        x[lhs] = x[other];
        x[other] = value;
      }
    }
    rund_numeric_sync();
  }
  if (control[0] != 0u) { return; }
  for (ulong row = 0ul; row < n; ++row) {
    for (ulong col = ulong(lane); col < rc; col += ulong(lanes)) {
      RUND_SCALAR sum = x[RUND_INDEX(row, col, n, rc, p.layout)];
      for (ulong k = 0ul; k < row; ++k) {
        sum = RUND_SUB(
            sum,
            RUND_MUL(factor[RUND_INDEX(row, k, n, n, p.layout)],
                     x[RUND_INDEX(k, col, n, rc, p.layout)]));
      }
      x[RUND_INDEX(row, col, n, rc, p.layout)] = sum;
    }
    rund_numeric_sync();
  }
  for (ulong reverse = 0ul; reverse < n; ++reverse) {
    ulong row = n - 1ul - reverse;
    RUND_SCALAR diagonal = factor[RUND_INDEX(row, row, n, n, p.layout)];
    if (lane == 0u && diagonal == RUND_ZERO) { control[0] = 1u; }
    rund_numeric_sync();
    if (control[0] != 0u) { break; }
    for (ulong col = ulong(lane); col < rc; col += ulong(lanes)) {
      RUND_SCALAR sum = x[RUND_INDEX(row, col, n, rc, p.layout)];
      for (ulong k = row + 1ul; k < n; ++k) {
        sum = RUND_SUB(
            sum,
            RUND_MUL(factor[RUND_INDEX(row, k, n, n, p.layout)],
                     x[RUND_INDEX(k, col, n, rc, p.layout)]));
      }
      x[RUND_INDEX(row, col, n, rc, p.layout)] = RUND_DIV(sum, diagonal);
    }
    rund_numeric_sync();
  }
}

inline void rund_solve_factor_cholesky(
    device const RUND_SCALAR* factor, device const RUND_SCALAR* rhs,
    device RUND_SCALAR* output, constant NumericParams& p, ulong batch,
    uint lane, uint lanes, threadgroup uint* control) {
  ulong n = p.rows;
  ulong rc = p.rhs_cols;
  device RUND_SCALAR* x = output + batch * n * rc;
  device const RUND_SCALAR* b = rhs + batch * n * rc;
  if (lane == 0u) { control[0] = 0u; }
  rund_numeric_sync();
  for (ulong row = 0ul; row < n; ++row) {
    RUND_SCALAR diagonal = factor[RUND_INDEX(row, row, n, n, p.layout)];
    if (lane == 0u && diagonal == RUND_ZERO) { control[0] = 2u; }
    rund_numeric_sync();
    if (control[0] != 0u) { break; }
    for (ulong col = ulong(lane); col < rc; col += ulong(lanes)) {
      RUND_SCALAR sum = b[RUND_INDEX(row, col, n, rc, p.layout)];
      for (ulong k = 0ul; k < row; ++k) {
        sum = RUND_SUB(
            sum,
            RUND_MUL(factor[RUND_INDEX(row, k, n, n, p.layout)],
                     x[RUND_INDEX(k, col, n, rc, p.layout)]));
      }
      x[RUND_INDEX(row, col, n, rc, p.layout)] = RUND_DIV(sum, diagonal);
    }
    rund_numeric_sync();
  }
  if (control[0] != 0u) { return; }
  for (ulong reverse = 0ul; reverse < n; ++reverse) {
    ulong row = n - 1ul - reverse;
    RUND_SCALAR diagonal = factor[RUND_INDEX(row, row, n, n, p.layout)];
    if (lane == 0u && diagonal == RUND_ZERO) { control[0] = 2u; }
    rund_numeric_sync();
    if (control[0] != 0u) { break; }
    for (ulong col = ulong(lane); col < rc; col += ulong(lanes)) {
      RUND_SCALAR sum = x[RUND_INDEX(row, col, n, rc, p.layout)];
      for (ulong k = row + 1ul; k < n; ++k) {
        sum = RUND_SUB(
            sum,
            RUND_MUL(factor[RUND_INDEX(k, row, n, n, p.layout)],
                     x[RUND_INDEX(k, col, n, rc, p.layout)]));
      }
      x[RUND_INDEX(row, col, n, rc, p.layout)] = RUND_DIV(sum, diagonal);
    }
    rund_numeric_sync();
  }
}

inline void rund_solve_factor_qr(
    device const RUND_SCALAR* factor, device const RUND_SCALAR* rhs,
    device RUND_SCALAR* output, constant NumericParams& p, ulong batch,
    uint lane, uint lanes, threadgroup uint* control) {
  ulong n = p.rows;
  ulong rc = p.rhs_cols;
  device const RUND_SCALAR* q = factor;
  device const RUND_SCALAR* r = factor + n * n;
  device const RUND_SCALAR* b = rhs + batch * n * rc;
  device RUND_SCALAR* x = output + batch * n * rc;
  if (lane == 0u) { control[0] = n > 16ul || rc > 16ul ? 4u : 0u; }
  rund_numeric_sync();
  if (control[0] != 0u) { return; }
  for (ulong cell = ulong(lane); cell < n * rc; cell += ulong(lanes)) {
    ulong row = cell / rc;
    ulong col = cell % rc;
    RUND_SCALAR sum = RUND_ZERO;
    for (ulong k = 0ul; k < n; ++k) {
      sum = RUND_ADD(
          sum,
          RUND_MUL(q[RUND_INDEX(k, row, n, n, p.layout)],
                   b[RUND_INDEX(k, col, n, rc, p.layout)]));
    }
    x[RUND_INDEX(row, col, n, rc, p.layout)] = sum;
  }
  rund_numeric_sync();
  for (ulong reverse = 0ul; reverse < n; ++reverse) {
    ulong row = n - 1ul - reverse;
    RUND_SCALAR diagonal = r[RUND_INDEX(row, row, n, n, p.layout)];
    if (lane == 0u && diagonal == RUND_ZERO) { control[0] = 1u; }
    rund_numeric_sync();
    if (control[0] != 0u) { break; }
    for (ulong col = ulong(lane); col < rc; col += ulong(lanes)) {
      RUND_SCALAR sum = x[RUND_INDEX(row, col, n, rc, p.layout)];
      for (ulong k = row + 1ul; k < n; ++k) {
        sum = RUND_SUB(
            sum,
            RUND_MUL(r[RUND_INDEX(row, k, n, n, p.layout)],
                     x[RUND_INDEX(k, col, n, rc, p.layout)]));
      }
      x[RUND_INDEX(row, col, n, rc, p.layout)] = RUND_DIV(sum, diagonal);
    }
    rund_numeric_sync();
  }
}

)MSL";

} // namespace rund::node::accel::detail::source::program::solve
