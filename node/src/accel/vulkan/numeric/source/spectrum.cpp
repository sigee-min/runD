#include "../source.hpp"

#include <string>

namespace rund::node::accel::detail {

[[nodiscard]] std::string SpectrumSource() {
  return NumericBaseSource() + R"GLSL(
layout(local_size_x = 32) in;
layout(set = 0, binding = 1, std430) readonly buffer Input { int input_values[]; };
layout(set = 0, binding = 2, std430) buffer Values { int value_values[]; };
layout(set = 0, binding = 3, std430) buffer Vectors { int vector_values[]; };
layout(set = 0, binding = 4, std430) buffer Status { uint status_values[]; };

shared int spectrum_a[256];
shared int spectrum_values[16];
shared int spectrum_vectors[256];
shared int spectrum_u[256];
shared int spectrum_c;
shared int spectrum_s;
shared int spectrum_dot;
shared int spectrum_norm;
shared uint spectrum_code;
shared uint spectrum_converged;
shared uint spectrum_p;
shared uint spectrum_q;
shared uint spectrum_swap;
shared uint spectrum_zero_norm;

void sync_spectrum() {
  memoryBarrierShared();
  memoryBarrierBuffer();
  barrier();
}

uint jacobi_values(uint n, bool track_vectors) {
  uint lane = gl_LocalInvocationID.x;
  if (lane == 0u) { spectrum_converged = 0u; }
  if (track_vectors) {
    for (uint cell = lane; cell < n * n; cell += 32u) {
      uint row = cell / n;
      uint col = cell % n;
      spectrum_vectors[cell] = row == col ? one_q31() : 0;
    }
  }
  sync_spectrum();
  for (uint iter = 0u; iter < p.max_iterations; ++iter) {
    if (lane == 0u && spectrum_converged == 0u) {
      spectrum_p = 0u;
      spectrum_q = 1u;
      uint best = 0u;
      for (uint row = 0u; row < n; ++row) {
        for (uint col = row + 1u; col < n; ++col) {
          uint magnitude =
              RundAbsMagnitude32(uint(spectrum_a[row * n + col]));
          if (magnitude > best) {
            best = magnitude;
            spectrum_p = row;
            spectrum_q = col;
          }
        }
      }
      if (best <= epsilon_q31()) {
        spectrum_converged = 1u;
      } else {
        int app = spectrum_a[spectrum_p * n + spectrum_p];
        int aqq = spectrum_a[spectrum_q * n + spectrum_q];
        int apq = spectrum_a[spectrum_p * n + spectrum_q];
        int tau = div_q31(sub_q31(aqq, app), add_q31(apq, apq));
        int quarter = one_q31() / 4;
        int sixteenth = one_q31() / 16;
        int tau_scaled = mul_q31(tau, quarter);
        int root_scaled =
            sqrt_q31(add_q31(sixteenth, mul_q31(tau_scaled, tau_scaled)));
        int abs_tau = int(min(RundAbsMagnitude32(uint(tau)), 0x7fffffffu));
        int sign_scaled = tau < 0 ? -quarter : quarter;
        int t = div_q31(sign_scaled,
                        add_q31(mul_q31(abs_tau, quarter), root_scaled));
        int t_scaled = mul_q31(t, quarter);
        spectrum_c = div_q31(
            quarter,
            sqrt_q31(add_q31(sixteenth, mul_q31(t_scaled, t_scaled))));
        spectrum_s = mul_q31(t, spectrum_c);
      }
    }
    sync_spectrum();
    if (spectrum_converged == 0u) {
      for (uint k = lane; k < n; k += 32u) {
        int akp = spectrum_a[k * n + spectrum_p];
        int akq = spectrum_a[k * n + spectrum_q];
        spectrum_a[k * n + spectrum_p] =
            sub_q31(mul_q31(spectrum_c, akp),
                    mul_q31(spectrum_s, akq));
        spectrum_a[k * n + spectrum_q] =
            add_q31(mul_q31(spectrum_s, akp),
                    mul_q31(spectrum_c, akq));
      }
    }
    sync_spectrum();
    if (spectrum_converged == 0u) {
      for (uint k = lane; k < n; k += 32u) {
        int apk = spectrum_a[spectrum_p * n + k];
        int aqk = spectrum_a[spectrum_q * n + k];
        spectrum_a[spectrum_p * n + k] =
            sub_q31(mul_q31(spectrum_c, apk),
                    mul_q31(spectrum_s, aqk));
        spectrum_a[spectrum_q * n + k] =
            add_q31(mul_q31(spectrum_s, apk),
                    mul_q31(spectrum_c, aqk));
      }
    }
    sync_spectrum();
    if (track_vectors && spectrum_converged == 0u) {
      for (uint k = lane; k < n; k += 32u) {
        int vkp = spectrum_vectors[k * n + spectrum_p];
        int vkq = spectrum_vectors[k * n + spectrum_q];
        spectrum_vectors[k * n + spectrum_p] =
            sub_q31(mul_q31(spectrum_c, vkp),
                    mul_q31(spectrum_s, vkq));
        spectrum_vectors[k * n + spectrum_q] =
            add_q31(mul_q31(spectrum_s, vkp),
                    mul_q31(spectrum_c, vkq));
      }
    }
    sync_spectrum();
  }
  for (uint index = lane; index < n; index += 32u) {
    spectrum_values[index] = spectrum_a[index * n + index];
  }
  sync_spectrum();
  if (lane == 0u) {
    spectrum_code = spectrum_converged != 0u ? 0u : 1u;
  }
  sync_spectrum();
  return spectrum_code;
}

void sort_desc(uint n, bool track_vectors) {
  uint lane = gl_LocalInvocationID.x;
  for (uint i = 0u; i < n; ++i) {
    for (uint j = i + 1u; j < n; ++j) {
      if (lane == 0u) {
        spectrum_swap = 0u;
        if (spectrum_values[j] > spectrum_values[i]) {
          int value = spectrum_values[i];
          spectrum_values[i] = spectrum_values[j];
          spectrum_values[j] = value;
          spectrum_swap = 1u;
        }
      }
      sync_spectrum();
      if (track_vectors && spectrum_swap != 0u) {
        for (uint row = lane; row < n; row += 32u) {
          int value = spectrum_vectors[row * n + i];
          spectrum_vectors[row * n + i] = spectrum_vectors[row * n + j];
          spectrum_vectors[row * n + j] = value;
        }
      }
      sync_spectrum();
    }
  }
}

uint spectrum_batch(uint64_t batch) {
  uint lane = gl_LocalInvocationID.x;
  if (lane == 0u) {
    spectrum_code = (p.rows > uint64_t(16) || p.cols > uint64_t(16))
                        ? 2u
                        : 0u;
  }
  sync_spectrum();
  if (spectrum_code != 0u) { return spectrum_code; }
  bool track_vectors = p.vector_count != uint64_t(0);
  uint64_t base = batch * p.rows * p.cols;
  uint64_t value_base = batch * p.value_count;
  if (p.op == uint64_t(2)) {
    uint n = uint(p.rows);
    for (uint cell = lane; cell < n * n; cell += 32u) {
      uint row = cell / n;
      uint col = cell % n;
      spectrum_a[cell] = input_values[
          ix(base + midx(row, col, p.rows, p.cols, p.storage_layout))];
    }
    sync_spectrum();
    uint code = jacobi_values(n, track_vectors);
    for (uint index = lane; index < n; index += 32u) {
      value_values[ix(value_base + uint64_t(index))] =
          spectrum_values[index];
    }
    if (track_vectors) {
      uint64_t vec_base = batch * p.vector_count;
      for (uint cell = lane; cell < n * n; cell += 32u) {
        uint row = cell / n;
        uint col = cell % n;
        vector_values[ix(vec_base + midx(row, col, p.rows, p.rows,
                                         p.storage_layout))] =
            spectrum_vectors[cell];
      }
    }
    sync_spectrum();
    return code;
  }
  uint n = uint(p.cols);
  for (uint cell = lane; cell < n * n; cell += 32u) {
    uint left_col = cell / n;
    uint right_col = cell % n;
    int sum = 0;
    for (uint row = 0u; row < uint(p.rows); ++row) {
      sum = add_q31(
          sum, mul_q31(
                   input_values[ix(base + midx(row, left_col, p.rows,
                                               p.cols, p.storage_layout))],
                   input_values[ix(base + midx(row, right_col, p.rows,
                                               p.cols, p.storage_layout))]));
    }
    spectrum_a[cell] = sum;
  }
  sync_spectrum();
  uint code = jacobi_values(n, track_vectors);
  for (uint index = lane; index < n; index += 32u) {
    spectrum_values[index] = spectrum_values[index] < 0
                                 ? 0
                                 : sqrt_q31(spectrum_values[index]);
  }
  sync_spectrum();
  sort_desc(n, track_vectors);
  uint width = uint(min(p.rows, p.cols));
  for (uint index = lane; index < width; index += 32u) {
    value_values[ix(value_base + uint64_t(index))] =
        spectrum_values[index];
  }
  if (track_vectors) {
    uint64_t vec_base = batch * p.vector_count;
    uint vector_cols = p.mode == 3u ? width : uint(p.rows);
    uint rows = uint(p.rows);
    for (uint cell = lane; cell < rows * vector_cols; cell += 32u) {
      uint row = cell / vector_cols;
      uint col = cell % vector_cols;
      int value = 0;
      if (col < width && RundAbsMagnitude32(
                               uint(spectrum_values[col])) > epsilon_q31()) {
        int sum = 0;
        for (uint k = 0u; k < uint(p.cols); ++k) {
          sum = add_q31(
              sum, mul_q31(
                       input_values[ix(base + midx(row, k, p.rows, p.cols,
                                                   p.storage_layout))],
                       spectrum_vectors[k * n + col]));
        }
        value = div_q31(sum, spectrum_values[col]);
      } else if (row == col) {
        value = one_q31();
      }
      spectrum_u[cell] = value;
    }
    sync_spectrum();
    for (uint col = 0u; col < vector_cols; ++col) {
      for (uint previous = 0u; previous < col; ++previous) {
        if (lane == 0u) {
          spectrum_dot = 0;
          for (uint row = 0u; row < rows; ++row) {
            spectrum_dot = add_q31(
                spectrum_dot,
                mul_q31(spectrum_u[row * vector_cols + previous],
                        spectrum_u[row * vector_cols + col]));
          }
        }
        sync_spectrum();
        for (uint row = lane; row < rows; row += 32u) {
          spectrum_u[row * vector_cols + col] = sub_q31(
              spectrum_u[row * vector_cols + col],
              mul_q31(spectrum_dot,
                      spectrum_u[row * vector_cols + previous]));
        }
        sync_spectrum();
      }
      if (lane == 0u) {
        int squared = 0;
        for (uint row = 0u; row < rows; ++row) {
          int value = spectrum_u[row * vector_cols + col];
          squared = add_q31(squared, mul_q31(value, value));
        }
        spectrum_norm = sqrt_q31(squared);
        spectrum_zero_norm = spectrum_norm == 0 ? 1u : 0u;
        if (spectrum_zero_norm != 0u) { spectrum_norm = one_q31(); }
      }
      sync_spectrum();
      for (uint row = lane; row < rows; row += 32u) {
        int value = spectrum_zero_norm != 0u
                        ? (row == (col % rows) ? one_q31() : 0)
                        : spectrum_u[row * vector_cols + col];
        spectrum_u[row * vector_cols + col] =
            RundAbsMagnitude32(uint(value)) ==
                    RundAbsMagnitude32(uint(spectrum_norm))
                ? ((value < 0) != (spectrum_norm < 0)
                       ? int(0x80000000u)
                       : one_q31())
                : div_q31(value, spectrum_norm);
      }
      sync_spectrum();
    }
    for (uint cell = lane; cell < rows * vector_cols; cell += 32u) {
      uint row = cell / vector_cols;
      uint col = cell % vector_cols;
      vector_values[ix(vec_base + midx(row, col, p.rows, vector_cols,
                                       p.storage_layout))] = spectrum_u[cell];
    }
    sync_spectrum();
  }
  return code;
}

void main() {
  uint64_t batch = uint64_t(gl_WorkGroupID.x);
  if (batch >= p.batch_count) { return; }
  uint code = spectrum_batch(batch);
  if (gl_LocalInvocationID.x == 0u) { status_values[ix(batch)] = code; }
}
)GLSL";
}

} // namespace rund::node::accel::detail
