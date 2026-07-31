#pragma once

#include <string_view>

namespace rund::node::accel::detail::source::program {

inline constexpr std::string_view Factor = R"MSL(inline void rund_factor_lu(
    device const RUND_SCALAR* input, device RUND_SCALAR* factor,
    device uint* aux, constant NumericParams& p, ulong batch, uint lane,
    uint lanes, threadgroup uint* control) {
  ulong n = p.rows;
  device RUND_SCALAR* a = factor + batch * n * n;
  device const RUND_SCALAR* src = input + batch * n * n;
  device uint* piv = aux + batch * n;
  for (ulong i = ulong(lane); i < n * n; i += ulong(lanes)) { a[i] = src[i]; }
  if (lane == 0u) { control[0] = 0u; }
  rund_numeric_sync();
  for (ulong k = 0ul; k < n; ++k) {
    if (lane == 0u) {
      ulong pivot = k;
      RUND_MAGNITUDE best = RUND_MAG_ZERO;
      if (p.aux == 2u) {
        for (ulong row = k; row < n; ++row) {
          RUND_MAGNITUDE magnitude =
              RUND_ABS(a[RUND_INDEX(row, k, n, n, p.layout)]);
          if (magnitude > best) { best = magnitude; pivot = row; }
        }
      }
      control[1] = uint(pivot);
      piv[k] = uint(pivot);
    }
    rund_numeric_sync();
    ulong pivot = ulong(control[1]);
    if (pivot != k) {
      for (ulong col = ulong(lane); col < n; col += ulong(lanes)) {
        ulong lhs = RUND_INDEX(k, col, n, n, p.layout);
        ulong rhs = RUND_INDEX(pivot, col, n, n, p.layout);
        RUND_SCALAR value = a[lhs];
        a[lhs] = a[rhs];
        a[rhs] = value;
      }
    }
    rund_numeric_sync();
    if (lane == 0u && a[RUND_INDEX(k, k, n, n, p.layout)] == RUND_ZERO) {
      control[0] = 1u;
    }
    rund_numeric_sync();
    if (control[0] != 0u) { break; }
    RUND_SCALAR diagonal = a[RUND_INDEX(k, k, n, n, p.layout)];
    for (ulong row = k + 1ul + ulong(lane); row < n;
         row += ulong(lanes)) {
      ulong index = RUND_INDEX(row, k, n, n, p.layout);
      a[index] = RUND_DIV(a[index], diagonal);
    }
    rund_numeric_sync();
    ulong width = n - k - 1ul;
    for (ulong cell = ulong(lane); cell < width * width;
         cell += ulong(lanes)) {
      ulong row = k + 1ul + cell / width;
      ulong col = k + 1ul + cell % width;
      ulong index = RUND_INDEX(row, col, n, n, p.layout);
      a[index] = RUND_SUB(
          a[index],
          RUND_MUL(a[RUND_INDEX(row, k, n, n, p.layout)],
                   a[RUND_INDEX(k, col, n, n, p.layout)]));
    }
    rund_numeric_sync();
  }
}

inline void rund_factor_cholesky(
    device const RUND_SCALAR* input, device RUND_SCALAR* factor,
    constant NumericParams& p, ulong batch, uint lane, uint lanes,
    threadgroup uint* control) {
  ulong n = p.rows;
  device RUND_SCALAR* l = factor + batch * n * n;
  device const RUND_SCALAR* a = input + batch * n * n;
  for (ulong i = ulong(lane); i < n * n; i += ulong(lanes)) { l[i] = RUND_ZERO; }
  if (lane == 0u) { control[0] = 0u; }
  rund_numeric_sync();
  // Column j depends only on columns [0,j).  Every below-diagonal row in the
  // column is independent and retains the reference's ascending k fold.
  for (ulong j = 0ul; j < n; ++j) {
    if (lane == 0u) {
      RUND_SCALAR sum = a[RUND_INDEX(j, j, n, n, p.layout)];
      for (ulong k = 0ul; k < j; ++k) {
        sum = RUND_SUB(sum,
                       RUND_MUL(l[RUND_INDEX(j, k, n, n, p.layout)],
                                l[RUND_INDEX(j, k, n, n, p.layout)]));
      }
      if (sum <= RUND_ZERO) {
        control[0] = 2u;
      } else {
        l[RUND_INDEX(j, j, n, n, p.layout)] = RUND_SQRT(sum);
      }
    }
    rund_numeric_sync();
    if (control[0] != 0u) { break; }
    RUND_SCALAR diagonal = l[RUND_INDEX(j, j, n, n, p.layout)];
    for (ulong row = j + 1ul + ulong(lane); row < n;
         row += ulong(lanes)) {
      RUND_SCALAR sum = a[RUND_INDEX(row, j, n, n, p.layout)];
      for (ulong k = 0ul; k < j; ++k) {
        sum = RUND_SUB(sum,
                       RUND_MUL(l[RUND_INDEX(row, k, n, n, p.layout)],
                                l[RUND_INDEX(j, k, n, n, p.layout)]));
      }
      l[RUND_INDEX(row, j, n, n, p.layout)] = RUND_DIV(sum, diagonal);
    }
    rund_numeric_sync();
  }
}

inline void rund_factor_qr(
    device const RUND_SCALAR* input, device RUND_SCALAR* factor,
    constant NumericParams& p, ulong batch, uint lane, uint lanes,
    threadgroup RUND_SCALAR* q, threadgroup RUND_SCALAR* r,
    threadgroup RUND_SCALAR* v, threadgroup RUND_SCALAR* scalar,
    threadgroup uint* control) {
  ulong rows = p.rows;
  ulong cols = p.cols;
  device const RUND_SCALAR* src = input + batch * rows * cols;
  for (ulong i = ulong(lane); i < 256ul; i += ulong(lanes)) {
    q[i] = RUND_ZERO;
    r[i] = RUND_ZERO;
  }
  if (lane == 0u) { control[0] = rows > 16ul || cols > 16ul ? 4u : 0u; }
  rund_numeric_sync();
  if (control[0] == 0u) {
    for (ulong col = 0ul; col < cols; ++col) {
      for (ulong row = ulong(lane); row < rows; row += ulong(lanes)) {
        v[row] = src[RUND_INDEX(row, col, rows, cols, p.layout)];
      }
      rund_numeric_sync();
      for (ulong previous = 0ul; previous < col; ++previous) {
        if (lane == 0u) {
          RUND_SCALAR dot = RUND_ZERO;
          for (ulong row = 0ul; row < rows; ++row) {
            dot = RUND_ADD(dot, RUND_MUL(q[row * cols + previous], v[row]));
          }
          scalar[0] = dot;
          r[previous * cols + col] = dot;
        }
        rund_numeric_sync();
        RUND_SCALAR dot = scalar[0];
        for (ulong row = ulong(lane); row < rows; row += ulong(lanes)) {
          v[row] = RUND_SUB(v[row], RUND_MUL(dot, q[row * cols + previous]));
        }
        rund_numeric_sync();
      }
      if (lane == 0u) {
        RUND_SCALAR square = RUND_ZERO;
        for (ulong row = 0ul; row < rows; ++row) {
          square = RUND_ADD(square, RUND_MUL(v[row], v[row]));
        }
        scalar[0] = RUND_SQRT(square);
        if (scalar[0] == RUND_ZERO) { control[0] = 1u; }
        r[col * cols + col] = scalar[0];
      }
      rund_numeric_sync();
      if (control[0] != 0u) { break; }
      RUND_SCALAR norm = scalar[0];
      for (ulong row = ulong(lane); row < rows; row += ulong(lanes)) {
        q[row * cols + col] = RUND_DIV(v[row], norm);
      }
      rund_numeric_sync();
    }
  }
  if (control[0] != 0u) { return; }
  device RUND_SCALAR* out = factor + batch * p.value_count;
  for (ulong cell = ulong(lane); cell < rows * cols; cell += ulong(lanes)) {
    ulong row = cell / cols;
    ulong col = cell % cols;
    out[RUND_INDEX(row, col, rows, cols, p.layout)] = q[cell];
  }
  if (p.mode == 2u) {
    device RUND_SCALAR* rout = out + rows * cols;
    for (ulong cell = ulong(lane); cell < cols * cols; cell += ulong(lanes)) {
      ulong row = cell / cols;
      ulong col = cell % cols;
      rout[RUND_INDEX(row, col, cols, cols, p.layout)] = r[cell];
    }
  }
  rund_numeric_sync();
}

kernel void RUND_KERNEL(rund_numeric_factor_)(
    device const RUND_SCALAR* input [[buffer(0)]],
    device RUND_SCALAR* factor [[buffer(1)]],
    device uint* aux [[buffer(2)]],
    device uint* status [[buffer(3)]],
    constant NumericParams& p [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 group_size [[threads_per_threadgroup]]) {
  ulong batch = ulong(group.x);
  if (batch >= p.batch_count) { return; }
  threadgroup RUND_SCALAR q[256];
  threadgroup RUND_SCALAR r[256];
  threadgroup RUND_SCALAR v[16];
  threadgroup RUND_SCALAR scalar[2];
  threadgroup uint control[4];
  uint lanes = group_size.x;
  if (p.op == 1ul) {
    rund_factor_lu(input, factor, aux, p, batch, lane, lanes, control);
  } else if (p.op == 3ul) {
    rund_factor_cholesky(input, factor, p, batch, lane, lanes, control);
  } else {
    rund_factor_qr(input, factor, p, batch, lane, lanes, q, r, v, scalar,
                   control);
  }
  if (lane == 0u) { status[batch] = control[0]; }
}

)MSL";

} // namespace rund::node::accel::detail::source::program
