#pragma once

#include <string_view>

namespace rund::node::accel::detail::source::program {

inline constexpr std::string_view Spectrum = R"MSL(inline void rund_jacobi(
    threadgroup RUND_SCALAR* a, ulong n, uint max_iterations,
    threadgroup RUND_SCALAR* values, threadgroup RUND_SCALAR* vectors,
    bool track_vectors, constant NumericParams& p, uint lane, uint lanes,
    threadgroup RUND_SCALAR* scalar, threadgroup uint* control) {
  if (track_vectors) {
    for (ulong cell = ulong(lane); cell < n * n; cell += ulong(lanes)) {
      ulong row = cell / n;
      ulong col = cell % n;
      vectors[cell] = row == col ? RUND_ONE() : RUND_ZERO;
    }
  }
  if (lane == 0u) { control[3] = 0u; }
  rund_numeric_sync();
  for (uint iteration = 0u; iteration < max_iterations; ++iteration) {
    if (lane == 0u) {
      ulong pivot_row = 0ul;
      ulong pivot_col = 1ul;
      RUND_MAGNITUDE best = RUND_MAG_ZERO;
      for (ulong row = 0ul; row < n; ++row) {
        for (ulong col = row + 1ul; col < n; ++col) {
          RUND_MAGNITUDE magnitude = RUND_ABS(a[row * n + col]);
          if (magnitude > best) {
            best = magnitude;
            pivot_row = row;
            pivot_col = col;
          }
        }
      }
      control[0] = uint(best <= RUND_EPSILON(p));
      control[1] = uint(pivot_row);
      control[2] = uint(pivot_col);
      if (control[0] != 0u) { control[3] = 1u; }
    }
    rund_numeric_sync();
    if (control[0] != 0u) { break; }
    ulong pivot_row = ulong(control[1]);
    ulong pivot_col = ulong(control[2]);
    if (lane == 0u) {
      RUND_SCALAR app = a[pivot_row * n + pivot_row];
      RUND_SCALAR aqq = a[pivot_col * n + pivot_col];
      RUND_SCALAR apq = a[pivot_row * n + pivot_col];
      RUND_SCALAR tau =
          RUND_DIV(RUND_SUB(aqq, app), RUND_ADD(apq, apq));
      RUND_SCALAR quarter = RUND_ONE() / RUND_FROM_ULONG(4ul);
      RUND_SCALAR sixteenth = RUND_ONE() / RUND_FROM_ULONG(16ul);
      RUND_SCALAR scaled_tau = RUND_MUL(tau, quarter);
      RUND_SCALAR scaled_root = RUND_SQRT(
          RUND_ADD(sixteenth, RUND_MUL(scaled_tau, scaled_tau)));
      RUND_SCALAR absolute_tau = RUND_CLAMP_ABS(tau);
      RUND_SCALAR sign = tau < RUND_ZERO ? -quarter : quarter;
      RUND_SCALAR t = RUND_DIV(
          sign, RUND_ADD(RUND_MUL(absolute_tau, quarter), scaled_root));
      RUND_SCALAR scaled_t = RUND_MUL(t, quarter);
      RUND_SCALAR c = RUND_DIV(
          quarter,
          RUND_SQRT(RUND_ADD(sixteenth, RUND_MUL(scaled_t, scaled_t))));
      scalar[0] = c;
      scalar[1] = RUND_MUL(t, c);
    }
    rund_numeric_sync();
    RUND_SCALAR c = scalar[0];
    RUND_SCALAR s = scalar[1];
    for (ulong k = ulong(lane); k < n; k += ulong(lanes)) {
      RUND_SCALAR akp = a[k * n + pivot_row];
      RUND_SCALAR akq = a[k * n + pivot_col];
      a[k * n + pivot_row] =
          RUND_SUB(RUND_MUL(c, akp), RUND_MUL(s, akq));
      a[k * n + pivot_col] =
          RUND_ADD(RUND_MUL(s, akp), RUND_MUL(c, akq));
    }
    rund_numeric_sync();
    for (ulong k = ulong(lane); k < n; k += ulong(lanes)) {
      RUND_SCALAR apk = a[pivot_row * n + k];
      RUND_SCALAR aqk = a[pivot_col * n + k];
      a[pivot_row * n + k] =
          RUND_SUB(RUND_MUL(c, apk), RUND_MUL(s, aqk));
      a[pivot_col * n + k] =
          RUND_ADD(RUND_MUL(s, apk), RUND_MUL(c, aqk));
    }
    rund_numeric_sync();
    if (track_vectors) {
      for (ulong k = ulong(lane); k < n; k += ulong(lanes)) {
        RUND_SCALAR vkp = vectors[k * n + pivot_row];
        RUND_SCALAR vkq = vectors[k * n + pivot_col];
        vectors[k * n + pivot_row] =
            RUND_SUB(RUND_MUL(c, vkp), RUND_MUL(s, vkq));
        vectors[k * n + pivot_col] =
            RUND_ADD(RUND_MUL(s, vkp), RUND_MUL(c, vkq));
      }
    }
    rund_numeric_sync();
  }
  for (ulong i = ulong(lane); i < n; i += ulong(lanes)) {
    values[i] = a[i * n + i];
  }
  rund_numeric_sync();
}

inline void rund_sort_spectrum(
    threadgroup RUND_SCALAR* values, threadgroup RUND_SCALAR* vectors,
    ulong n, bool track_vectors, uint lane, uint lanes,
    threadgroup uint* control) {
  for (ulong i = 0ul; i < n; ++i) {
    for (ulong j = i + 1ul; j < n; ++j) {
      if (lane == 0u) {
        control[0] = uint(values[j] > values[i]);
        if (control[0] != 0u) {
          RUND_SCALAR value = values[i];
          values[i] = values[j];
          values[j] = value;
        }
      }
      rund_numeric_sync();
      if (track_vectors && control[0] != 0u) {
        for (ulong row = ulong(lane); row < n; row += ulong(lanes)) {
          RUND_SCALAR value = vectors[row * n + i];
          vectors[row * n + i] = vectors[row * n + j];
          vectors[row * n + j] = value;
        }
      }
      rund_numeric_sync();
    }
  }
}

kernel void RUND_KERNEL(rund_numeric_spectrum_)(
    device const RUND_SCALAR* input [[buffer(0)]],
    device RUND_SCALAR* values [[buffer(1)]],
    device RUND_SCALAR* vectors [[buffer(2)]],
    device uint* status [[buffer(3)]],
    constant NumericParams& p [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 group_size [[threads_per_threadgroup]]) {
  ulong batch = ulong(group.x);
  if (batch >= p.batch_count) { return; }
  threadgroup RUND_SCALAR a[256];
  threadgroup RUND_SCALAR vals[16];
  threadgroup RUND_SCALAR vecs[256];
  threadgroup RUND_SCALAR u[256];
  threadgroup RUND_SCALAR scalar[4];
  threadgroup uint control[8];
  uint lanes = group_size.x;
  if (lane == 0u) {
    control[4] = p.rows > 16ul || p.cols > 16ul ? 2u : 0u;
  }
  rund_numeric_sync();
  if (control[4] != 0u) {
    if (lane == 0u) { status[batch] = control[4]; }
    return;
  }
  bool track_vectors = p.vector_count != 0ul;
  device const RUND_SCALAR* src = input + batch * p.rows * p.cols;
  if (p.op == 2ul) {
    for (ulong cell = ulong(lane); cell < p.rows * p.rows;
         cell += ulong(lanes)) {
      ulong row = cell / p.rows;
      ulong col = cell % p.rows;
      a[cell] = src[RUND_INDEX(row, col, p.rows, p.cols, p.layout)];
    }
    rund_numeric_sync();
    rund_jacobi(a, p.rows, p.max_iterations, vals, vecs, track_vectors, p,
                 lane, lanes, scalar, control);
    device RUND_SCALAR* out = values + batch * p.value_count;
    for (ulong i = ulong(lane); i < p.rows; i += ulong(lanes)) {
      out[i] = vals[i];
    }
    if (track_vectors) {
      device RUND_SCALAR* out_vectors = vectors + batch * p.vector_count;
      for (ulong cell = ulong(lane); cell < p.rows * p.rows;
           cell += ulong(lanes)) {
        ulong row = cell / p.rows;
        ulong col = cell % p.rows;
        out_vectors[RUND_INDEX(row, col, p.rows, p.rows, p.layout)] =
            vecs[cell];
      }
    }
    if (lane == 0u) { status[batch] = control[3] != 0u ? 0u : 1u; }
    return;
  }
  for (ulong cell = ulong(lane); cell < p.cols * p.cols;
       cell += ulong(lanes)) {
    ulong lhs = cell / p.cols;
    ulong rhs_col = cell % p.cols;
    RUND_SCALAR sum = RUND_ZERO;
    for (ulong row = 0ul; row < p.rows; ++row) {
      sum = RUND_ADD(
          sum,
          RUND_MUL(src[RUND_INDEX(row, lhs, p.rows, p.cols, p.layout)],
                   src[RUND_INDEX(row, rhs_col, p.rows, p.cols, p.layout)]));
    }
    a[cell] = sum;
  }
  rund_numeric_sync();
  rund_jacobi(a, p.cols, p.max_iterations, vals, vecs, track_vectors, p,
               lane, lanes, scalar, control);
  for (ulong i = ulong(lane); i < p.cols; i += ulong(lanes)) {
    vals[i] = vals[i] < RUND_ZERO ? RUND_ZERO : RUND_SQRT(vals[i]);
  }
  rund_numeric_sync();
  rund_sort_spectrum(vals, vecs, p.cols, track_vectors, lane, lanes, control);
  ulong width = min(p.rows, p.cols);
  device RUND_SCALAR* out = values + batch * p.value_count;
  for (ulong i = ulong(lane); i < width; i += ulong(lanes)) { out[i] = vals[i]; }
  if (track_vectors) {
    device RUND_SCALAR* out_vectors = vectors + batch * p.vector_count;
    ulong vector_cols = p.mode == 3u ? width : p.rows;
    for (ulong cell = ulong(lane); cell < p.rows * vector_cols;
         cell += ulong(lanes)) {
      ulong row = cell / vector_cols;
      ulong col = cell % vector_cols;
      RUND_SCALAR value = RUND_ZERO;
      if (col < width && RUND_ABS(vals[col]) > RUND_EPSILON(p)) {
        RUND_SCALAR sum = RUND_ZERO;
        for (ulong k = 0ul; k < p.cols; ++k) {
          sum = RUND_ADD(
              sum,
              RUND_MUL(src[RUND_INDEX(row, k, p.rows, p.cols, p.layout)],
                       vecs[k * p.cols + col]));
        }
        value = RUND_DIV(sum, vals[col]);
      } else if (row == col) {
        value = RUND_ONE();
      }
      u[cell] = value;
    }
    rund_numeric_sync();
    for (ulong col = 0ul; col < vector_cols; ++col) {
      for (ulong previous = 0ul; previous < col; ++previous) {
        if (lane == 0u) {
          RUND_SCALAR dot = RUND_ZERO;
          for (ulong row = 0ul; row < p.rows; ++row) {
            dot = RUND_ADD(dot,
                           RUND_MUL(u[row * vector_cols + previous],
                                    u[row * vector_cols + col]));
          }
          scalar[0] = dot;
        }
        rund_numeric_sync();
        RUND_SCALAR dot = scalar[0];
        for (ulong row = ulong(lane); row < p.rows; row += ulong(lanes)) {
          ulong index = row * vector_cols + col;
          u[index] = RUND_SUB(
              u[index], RUND_MUL(dot, u[row * vector_cols + previous]));
        }
        rund_numeric_sync();
      }
      if (lane == 0u) {
        RUND_SCALAR square = RUND_ZERO;
        for (ulong row = 0ul; row < p.rows; ++row) {
          RUND_SCALAR value = u[row * vector_cols + col];
          square = RUND_ADD(square, RUND_MUL(value, value));
        }
        scalar[0] = RUND_SQRT(square);
        control[0] = uint(scalar[0] == RUND_ZERO);
        if (control[0] != 0u) { scalar[0] = RUND_ONE(); }
      }
      rund_numeric_sync();
      if (control[0] != 0u) {
        for (ulong row = ulong(lane); row < p.rows; row += ulong(lanes)) {
          u[row * vector_cols + col] =
              row == (col % p.rows) ? RUND_ONE() : RUND_ZERO;
        }
      }
      rund_numeric_sync();
      RUND_SCALAR norm = scalar[0];
      for (ulong row = ulong(lane); row < p.rows; row += ulong(lanes)) {
        ulong index = row * vector_cols + col;
        RUND_SCALAR value = u[index];
        u[index] = RUND_ABS(value) == RUND_ABS(norm)
                       ? ((value < RUND_ZERO) != (norm < RUND_ZERO)
                              ? RUND_NEGATIVE_ONE
                              : RUND_ONE())
                       : RUND_DIV(value, norm);
      }
      rund_numeric_sync();
    }
    for (ulong cell = ulong(lane); cell < p.rows * vector_cols;
         cell += ulong(lanes)) {
      ulong row = cell / vector_cols;
      ulong col = cell % vector_cols;
      out_vectors[RUND_INDEX(row, col, p.rows, vector_cols, p.layout)] =
          u[cell];
    }
  }
  if (lane == 0u) { status[batch] = control[3] != 0u ? 0u : 1u; }
}

)MSL";

} // namespace rund::node::accel::detail::source::program
