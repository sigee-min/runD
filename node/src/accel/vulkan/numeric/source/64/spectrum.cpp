#include "../../source.hpp"

#include <string>

namespace rund::node::accel::detail {

[[nodiscard]] std::string SpectrumSource64() {
  return NumericBaseSource64() + R"GLSL(
layout(local_size_x = 32) in;
layout(set = 0, binding = 1, std430) readonly buffer Input { int64_t input_values[]; };
layout(set = 0, binding = 2, std430) buffer Values { int64_t value_values[]; };
layout(set = 0, binding = 3, std430) buffer Vectors { int64_t vector_values[]; };
layout(set = 0, binding = 4, std430) buffer Status { uint status_values[]; };

shared int64_t spectrum_a64[256];
shared int64_t spectrum_values64[16];
shared int64_t spectrum_vectors64[256];
shared int64_t spectrum_u64[256];
shared int64_t spectrum_c64;
shared int64_t spectrum_s64;
shared int64_t spectrum_dot64;
shared int64_t spectrum_norm64;
shared uint spectrum_code64;
shared uint spectrum_converged64;
shared uint spectrum_p64;
shared uint spectrum_q64;
shared uint spectrum_swap64;
shared uint spectrum_zero_norm64;

void sync_spectrum64() {
  memoryBarrierShared();
  memoryBarrierBuffer();
  barrier();
}

uint jacobi_values64(uint n, bool track_vectors) {
  uint lane = gl_LocalInvocationID.x;
  if (lane == 0u) { spectrum_converged64 = 0u; }
  if (track_vectors) {
    for (uint cell = lane; cell < n * n; cell += 32u) {
      uint row = cell / n;
      uint col = cell % n;
      spectrum_vectors64[cell] =
          row == col ? one_q63() : int64_t(0);
    }
  }
  sync_spectrum64();
  for (uint iter = 0u; iter < p.max_iterations; ++iter) {
    if (lane == 0u && spectrum_converged64 == 0u) {
      spectrum_p64 = 0u;
      spectrum_q64 = 1u;
      uint64_t best = uint64_t(0);
      for (uint row = 0u; row < n; ++row) {
        for (uint col = row + 1u; col < n; ++col) {
          uint64_t magnitude = RundAbsMagnitude64(
              as_u64(spectrum_a64[row * n + col]));
          if (magnitude > best) {
            best = magnitude;
            spectrum_p64 = row;
            spectrum_q64 = col;
          }
        }
      }
      if (best <= epsilon_q63()) {
        spectrum_converged64 = 1u;
      } else {
        int64_t app = spectrum_a64[spectrum_p64 * n + spectrum_p64];
        int64_t aqq = spectrum_a64[spectrum_q64 * n + spectrum_q64];
        int64_t apq = spectrum_a64[spectrum_p64 * n + spectrum_q64];
        int64_t tau = div_q63(sub_q63(aqq, app), add_q63(apq, apq));
        int64_t quarter = one_q63() / int64_t(4);
        int64_t sixteenth = one_q63() / int64_t(16);
        int64_t tau_scaled = mul_q63(tau, quarter);
        int64_t root_scaled =
            sqrt_q63(add_q63(sixteenth, mul_q63(tau_scaled, tau_scaled)));
        int64_t abs_tau = as_i64(min(RundAbsMagnitude64(as_u64(tau)),
                                     0x7ffffffffffffffful));
        int64_t sign_scaled = tau < int64_t(0) ? -quarter : quarter;
        int64_t t = div_q63(
            sign_scaled, add_q63(mul_q63(abs_tau, quarter), root_scaled));
        int64_t t_scaled = mul_q63(t, quarter);
        spectrum_c64 = div_q63(
            quarter,
            sqrt_q63(add_q63(sixteenth, mul_q63(t_scaled, t_scaled))));
        spectrum_s64 = mul_q63(t, spectrum_c64);
      }
    }
    sync_spectrum64();
    if (spectrum_converged64 == 0u) {
      for (uint k = lane; k < n; k += 32u) {
        int64_t akp = spectrum_a64[k * n + spectrum_p64];
        int64_t akq = spectrum_a64[k * n + spectrum_q64];
        spectrum_a64[k * n + spectrum_p64] =
            sub_q63(mul_q63(spectrum_c64, akp),
                    mul_q63(spectrum_s64, akq));
        spectrum_a64[k * n + spectrum_q64] =
            add_q63(mul_q63(spectrum_s64, akp),
                    mul_q63(spectrum_c64, akq));
      }
    }
    sync_spectrum64();
    if (spectrum_converged64 == 0u) {
      for (uint k = lane; k < n; k += 32u) {
        int64_t apk = spectrum_a64[spectrum_p64 * n + k];
        int64_t aqk = spectrum_a64[spectrum_q64 * n + k];
        spectrum_a64[spectrum_p64 * n + k] =
            sub_q63(mul_q63(spectrum_c64, apk),
                    mul_q63(spectrum_s64, aqk));
        spectrum_a64[spectrum_q64 * n + k] =
            add_q63(mul_q63(spectrum_s64, apk),
                    mul_q63(spectrum_c64, aqk));
      }
    }
    sync_spectrum64();
    if (track_vectors && spectrum_converged64 == 0u) {
      for (uint k = lane; k < n; k += 32u) {
        int64_t vkp = spectrum_vectors64[k * n + spectrum_p64];
        int64_t vkq = spectrum_vectors64[k * n + spectrum_q64];
        spectrum_vectors64[k * n + spectrum_p64] =
            sub_q63(mul_q63(spectrum_c64, vkp),
                    mul_q63(spectrum_s64, vkq));
        spectrum_vectors64[k * n + spectrum_q64] =
            add_q63(mul_q63(spectrum_s64, vkp),
                    mul_q63(spectrum_c64, vkq));
      }
    }
    sync_spectrum64();
  }
  for (uint index = lane; index < n; index += 32u) {
    spectrum_values64[index] = spectrum_a64[index * n + index];
  }
  sync_spectrum64();
  if (lane == 0u) {
    spectrum_code64 = spectrum_converged64 != 0u ? 0u : 1u;
  }
  sync_spectrum64();
  return spectrum_code64;
}

void sort_desc64(uint n, bool track_vectors) {
  uint lane = gl_LocalInvocationID.x;
  for (uint i = 0u; i < n; ++i) {
    for (uint j = i + 1u; j < n; ++j) {
      if (lane == 0u) {
        spectrum_swap64 = 0u;
        if (spectrum_values64[j] > spectrum_values64[i]) {
          int64_t value = spectrum_values64[i];
          spectrum_values64[i] = spectrum_values64[j];
          spectrum_values64[j] = value;
          spectrum_swap64 = 1u;
        }
      }
      sync_spectrum64();
      if (track_vectors && spectrum_swap64 != 0u) {
        for (uint row = lane; row < n; row += 32u) {
          int64_t value = spectrum_vectors64[row * n + i];
          spectrum_vectors64[row * n + i] =
              spectrum_vectors64[row * n + j];
          spectrum_vectors64[row * n + j] = value;
        }
      }
      sync_spectrum64();
    }
  }
}

uint spectrum_batch64(uint64_t batch) {
  uint lane = gl_LocalInvocationID.x;
  if (lane == 0u) {
    spectrum_code64 = (p.rows > uint64_t(16) || p.cols > uint64_t(16))
                          ? 2u
                          : 0u;
  }
  sync_spectrum64();
  if (spectrum_code64 != 0u) { return spectrum_code64; }
  bool track_vectors = p.vector_count != uint64_t(0);
  uint64_t base = batch * p.rows * p.cols;
  uint64_t value_base = batch * p.value_count;
  if (p.op == uint64_t(2)) {
    uint n = uint(p.rows);
    for (uint cell = lane; cell < n * n; cell += 32u) {
      uint row = cell / n;
      uint col = cell % n;
      spectrum_a64[cell] = input_values[
          ix64(base + midx64(row, col, p.rows, p.cols, p.storage_layout))];
    }
    sync_spectrum64();
    uint code = jacobi_values64(n, track_vectors);
    for (uint index = lane; index < n; index += 32u) {
      value_values[ix64(value_base + uint64_t(index))] =
          spectrum_values64[index];
    }
    if (track_vectors) {
      uint64_t vec_base = batch * p.vector_count;
      for (uint cell = lane; cell < n * n; cell += 32u) {
        uint row = cell / n;
        uint col = cell % n;
        vector_values[ix64(vec_base + midx64(row, col, p.rows, p.rows,
                                             p.storage_layout))] =
            spectrum_vectors64[cell];
      }
    }
    sync_spectrum64();
    return code;
  }
  uint n = uint(p.cols);
  for (uint cell = lane; cell < n * n; cell += 32u) {
    uint left_col = cell / n;
    uint right_col = cell % n;
    int64_t sum = int64_t(0);
    for (uint row = 0u; row < uint(p.rows); ++row) {
      sum = add_q63(
          sum, mul_q63(
                   input_values[ix64(base + midx64(row, left_col, p.rows,
                                                   p.cols,
                                                   p.storage_layout))],
                   input_values[ix64(base + midx64(row, right_col, p.rows,
                                                   p.cols,
                                                   p.storage_layout))]));
    }
    spectrum_a64[cell] = sum;
  }
  sync_spectrum64();
  uint code = jacobi_values64(n, track_vectors);
  for (uint index = lane; index < n; index += 32u) {
    spectrum_values64[index] = spectrum_values64[index] < int64_t(0)
                                   ? int64_t(0)
                                   : sqrt_q63(spectrum_values64[index]);
  }
  sync_spectrum64();
  sort_desc64(n, track_vectors);
  uint width = uint(min(p.rows, p.cols));
  for (uint index = lane; index < width; index += 32u) {
    value_values[ix64(value_base + uint64_t(index))] =
        spectrum_values64[index];
  }
  if (track_vectors) {
    uint64_t vec_base = batch * p.vector_count;
    uint vector_cols = p.mode == 3u ? width : uint(p.rows);
    uint rows = uint(p.rows);
    for (uint cell = lane; cell < rows * vector_cols; cell += 32u) {
      uint row = cell / vector_cols;
      uint col = cell % vector_cols;
      int64_t value = int64_t(0);
      if (col < width && RundAbsMagnitude64(
                               as_u64(spectrum_values64[col])) >
                               epsilon_q63()) {
        int64_t sum = int64_t(0);
        for (uint k = 0u; k < uint(p.cols); ++k) {
          sum = add_q63(
              sum, mul_q63(
                       input_values[ix64(base + midx64(row, k, p.rows,
                                                       p.cols,
                                                       p.storage_layout))],
                       spectrum_vectors64[k * n + col]));
        }
        value = div_q63(sum, spectrum_values64[col]);
      } else if (row == col) {
        value = one_q63();
      }
      spectrum_u64[cell] = value;
    }
    sync_spectrum64();
    for (uint col = 0u; col < vector_cols; ++col) {
      for (uint previous = 0u; previous < col; ++previous) {
        if (lane == 0u) {
          spectrum_dot64 = int64_t(0);
          for (uint row = 0u; row < rows; ++row) {
            spectrum_dot64 = add_q63(
                spectrum_dot64,
                mul_q63(spectrum_u64[row * vector_cols + previous],
                        spectrum_u64[row * vector_cols + col]));
          }
        }
        sync_spectrum64();
        for (uint row = lane; row < rows; row += 32u) {
          spectrum_u64[row * vector_cols + col] = sub_q63(
              spectrum_u64[row * vector_cols + col],
              mul_q63(spectrum_dot64,
                      spectrum_u64[row * vector_cols + previous]));
        }
        sync_spectrum64();
      }
      if (lane == 0u) {
        int64_t squared = int64_t(0);
        for (uint row = 0u; row < rows; ++row) {
          int64_t value = spectrum_u64[row * vector_cols + col];
          squared = add_q63(squared, mul_q63(value, value));
        }
        spectrum_norm64 = sqrt_q63(squared);
        spectrum_zero_norm64 =
            spectrum_norm64 == int64_t(0) ? 1u : 0u;
        if (spectrum_zero_norm64 != 0u) {
          spectrum_norm64 = one_q63();
        }
      }
      sync_spectrum64();
      for (uint row = lane; row < rows; row += 32u) {
        int64_t value = spectrum_zero_norm64 != 0u
                            ? (row == (col % rows) ? one_q63()
                                                   : int64_t(0))
                            : spectrum_u64[row * vector_cols + col];
        spectrum_u64[row * vector_cols + col] =
            RundAbsMagnitude64(as_u64(value)) ==
                    RundAbsMagnitude64(as_u64(spectrum_norm64))
                ? ((value < int64_t(0)) != (spectrum_norm64 < int64_t(0))
                       ? as_i64(0x8000000000000000ul)
                       : one_q63())
                : div_q63(value, spectrum_norm64);
      }
      sync_spectrum64();
    }
    for (uint cell = lane; cell < rows * vector_cols; cell += 32u) {
      uint row = cell / vector_cols;
      uint col = cell % vector_cols;
      vector_values[ix64(vec_base + midx64(row, col, p.rows, vector_cols,
                                           p.storage_layout))] =
          spectrum_u64[cell];
    }
    sync_spectrum64();
  }
  return code;
}

void main() {
  uint64_t batch = uint64_t(gl_WorkGroupID.x);
  if (batch >= p.batch_count) { return; }
  uint code = spectrum_batch64(batch);
  if (gl_LocalInvocationID.x == 0u) { status_values[ix64(batch)] = code; }
}
)GLSL";
}

} // namespace rund::node::accel::detail
