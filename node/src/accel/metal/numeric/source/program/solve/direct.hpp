#pragma once

#include <string_view>

namespace rund::node::accel::detail::source::program::solve {

inline constexpr std::string_view Direct =
    R"MSL(inline void rund_solve_direct_lu(
    device const RUND_SCALAR* primary, device const RUND_SCALAR* rhs,
    device RUND_SCALAR* output, constant NumericParams& p, ulong batch,
    uint lane, uint lanes, threadgroup RUND_SCALAR* a,
    threadgroup RUND_SCALAR* x, threadgroup RUND_SCALAR* multipliers,
    threadgroup uint* control) {
  ulong n = p.rows;
  ulong rc = p.rhs_cols;
  device const RUND_SCALAR* src = primary + batch * n * n;
  device const RUND_SCALAR* b = rhs + batch * n * rc;
  for (ulong cell = ulong(lane); cell < n * n; cell += ulong(lanes)) {
    ulong row = cell / n;
    ulong col = cell % n;
    a[cell] = src[RUND_INDEX(row, col, n, n, p.layout)];
  }
  for (ulong cell = ulong(lane); cell < n * rc; cell += ulong(lanes)) {
    ulong row = cell / rc;
    ulong col = cell % rc;
    x[cell] = b[RUND_INDEX(row, col, n, rc, p.layout)];
  }
  if (lane == 0u) { control[0] = n > 16ul || rc > 16ul ? 4u : 0u; }
  rund_numeric_sync();
  for (ulong k = 0ul; k < n && control[0] == 0u; ++k) {
    if (lane == 0u) {
      ulong pivot = k;
      RUND_MAGNITUDE best = RUND_ABS(a[k * n + k]);
      for (ulong row = k + 1ul; row < n; ++row) {
        RUND_MAGNITUDE magnitude = RUND_ABS(a[row * n + k]);
        if (magnitude > best) { best = magnitude; pivot = row; }
      }
      control[1] = uint(pivot);
      if (best == RUND_MAG_ZERO) { control[0] = 1u; }
    }
    rund_numeric_sync();
    if (control[0] != 0u) { break; }
    ulong pivot = ulong(control[1]);
    if (pivot != k) {
      for (ulong col = ulong(lane); col < n; col += ulong(lanes)) {
        RUND_SCALAR value = a[k * n + col];
        a[k * n + col] = a[pivot * n + col];
        a[pivot * n + col] = value;
      }
      for (ulong col = ulong(lane); col < rc; col += ulong(lanes)) {
        RUND_SCALAR value = x[k * rc + col];
        x[k * rc + col] = x[pivot * rc + col];
        x[pivot * rc + col] = value;
      }
    }
    rund_numeric_sync();
    for (ulong row = k + 1ul + ulong(lane); row < n;
         row += ulong(lanes)) {
      multipliers[row] = RUND_DIV(a[row * n + k], a[k * n + k]);
      a[row * n + k] = RUND_ZERO;
    }
    rund_numeric_sync();
    ulong matrix_width = n - k - 1ul;
    for (ulong cell = ulong(lane); cell < matrix_width * matrix_width;
         cell += ulong(lanes)) {
      ulong row = k + 1ul + cell / matrix_width;
      ulong col = k + 1ul + cell % matrix_width;
      a[row * n + col] = RUND_SUB(
          a[row * n + col], RUND_MUL(multipliers[row], a[k * n + col]));
    }
    ulong rhs_cells = (n - k - 1ul) * rc;
    for (ulong cell = ulong(lane); cell < rhs_cells; cell += ulong(lanes)) {
      ulong row = k + 1ul + cell / rc;
      ulong col = cell % rc;
      x[row * rc + col] = RUND_SUB(
          x[row * rc + col], RUND_MUL(multipliers[row], x[k * rc + col]));
    }
    rund_numeric_sync();
  }
  if (control[0] != 0u) { return; }
  for (ulong reverse = 0ul; reverse < n; ++reverse) {
    ulong row = n - 1ul - reverse;
    RUND_SCALAR diagonal = a[row * n + row];
    if (lane == 0u && diagonal == RUND_ZERO) { control[0] = 1u; }
    rund_numeric_sync();
    if (control[0] != 0u) { break; }
    for (ulong col = ulong(lane); col < rc; col += ulong(lanes)) {
      RUND_SCALAR sum = x[row * rc + col];
      for (ulong k = row + 1ul; k < n; ++k) {
        sum = RUND_SUB(sum, RUND_MUL(a[row * n + k], x[k * rc + col]));
      }
      x[row * rc + col] = RUND_DIV(sum, diagonal);
    }
    rund_numeric_sync();
  }
  if (control[0] != 0u) { return; }
  device RUND_SCALAR* out = output + batch * n * rc;
  for (ulong cell = ulong(lane); cell < n * rc; cell += ulong(lanes)) {
    ulong row = cell / rc;
    ulong col = cell % rc;
    out[RUND_INDEX(row, col, n, rc, p.layout)] = x[cell];
  }
  rund_numeric_sync();
}

inline void rund_solve_direct_cholesky(
    device const RUND_SCALAR* primary, device const RUND_SCALAR* rhs,
    device RUND_SCALAR* output, constant NumericParams& p, ulong batch,
    uint lane, uint lanes, threadgroup RUND_SCALAR* l,
    threadgroup RUND_SCALAR* x, threadgroup uint* control) {
  ulong n = p.rows;
  ulong rc = p.rhs_cols;
  device const RUND_SCALAR* src = primary + batch * n * n;
  device const RUND_SCALAR* b = rhs + batch * n * rc;
  for (ulong i = ulong(lane); i < n * n; i += ulong(lanes)) { l[i] = RUND_ZERO; }
  if (lane == 0u) { control[0] = n > 16ul || rc > 16ul ? 4u : 0u; }
  rund_numeric_sync();
  for (ulong col = 0ul; col < n && control[0] == 0u; ++col) {
    if (lane == 0u) {
      RUND_SCALAR sum = src[RUND_INDEX(col, col, n, n, p.layout)];
      for (ulong k = 0ul; k < col; ++k) {
        sum = RUND_SUB(sum,
                       RUND_MUL(l[col * n + k], l[col * n + k]));
      }
      if (sum <= RUND_ZERO) {
        control[0] = 2u;
      } else {
        l[col * n + col] = RUND_SQRT(sum);
      }
    }
    rund_numeric_sync();
    if (control[0] != 0u) { break; }
    RUND_SCALAR diagonal = l[col * n + col];
    for (ulong row = col + 1ul + ulong(lane); row < n;
         row += ulong(lanes)) {
      RUND_SCALAR sum = src[RUND_INDEX(row, col, n, n, p.layout)];
      for (ulong k = 0ul; k < col; ++k) {
        sum = RUND_SUB(sum,
                       RUND_MUL(l[row * n + k], l[col * n + k]));
      }
      l[row * n + col] = RUND_DIV(sum, diagonal);
    }
    rund_numeric_sync();
  }
  if (control[0] != 0u) { return; }
  for (ulong row = 0ul; row < n; ++row) {
    RUND_SCALAR diagonal = l[row * n + row];
    for (ulong col = ulong(lane); col < rc; col += ulong(lanes)) {
      RUND_SCALAR sum = b[RUND_INDEX(row, col, n, rc, p.layout)];
      for (ulong k = 0ul; k < row; ++k) {
        sum = RUND_SUB(sum, RUND_MUL(l[row * n + k], x[k * rc + col]));
      }
      x[row * rc + col] = RUND_DIV(sum, diagonal);
    }
    rund_numeric_sync();
  }
  for (ulong reverse = 0ul; reverse < n; ++reverse) {
    ulong row = n - 1ul - reverse;
    RUND_SCALAR diagonal = l[row * n + row];
    for (ulong col = ulong(lane); col < rc; col += ulong(lanes)) {
      RUND_SCALAR sum = x[row * rc + col];
      for (ulong k = row + 1ul; k < n; ++k) {
        sum = RUND_SUB(sum, RUND_MUL(l[k * n + row], x[k * rc + col]));
      }
      x[row * rc + col] = RUND_DIV(sum, diagonal);
    }
    rund_numeric_sync();
  }
  device RUND_SCALAR* out = output + batch * n * rc;
  for (ulong cell = ulong(lane); cell < n * rc; cell += ulong(lanes)) {
    ulong row = cell / rc;
    ulong col = cell % rc;
    out[RUND_INDEX(row, col, n, rc, p.layout)] = x[cell];
  }
  rund_numeric_sync();
}

)MSL";

} // namespace rund::node::accel::detail::source::program::solve
