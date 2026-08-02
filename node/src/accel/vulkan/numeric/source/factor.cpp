#include "../source.hpp"

namespace rund::node::accel::detail {
namespace {

template <typename Sink>
[[nodiscard]] bool EmitFactorSource(Sink &sink)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  return EmitNumericBaseSource(sink, false) && sink.append(R"GLSL(
layout(local_size_x = 32) in;
layout(set = 0, binding = 1, std430) readonly buffer Input { int input_values[]; };
layout(set = 0, binding = 2, std430) buffer Factor { int factor_values[]; };
layout(set = 0, binding = 3, std430) buffer Aux { uint aux_values[]; };
layout(set = 0, binding = 4, std430) buffer Status { uint status_values[]; };

shared int factor_q[256];
shared int factor_r[256];
shared int factor_v[16];
shared int factor_diag;
shared int factor_dot;
shared int factor_norm;
shared uint factor_code;
shared uint factor_pivot;

void sync_factor() {
  memoryBarrierShared();
  memoryBarrierBuffer();
  barrier();
}

uint factor_lu(uint64_t batch) {
  uint lane = gl_LocalInvocationID.x;
  uint64_t n = p.rows;
  uint64_t base = batch * n * n;
  if (lane == 0u) { factor_code = 0u; }
  sync_factor();
  for (uint64_t i = uint64_t(lane); i < n * n; i += uint64_t(32)) {
    factor_values[ix(base + i)] = input_values[ix(base + i)];
  }
  sync_factor();
  for (uint64_t k = uint64_t(0); k < n; ++k) {
    if (lane == 0u && factor_code == 0u) {
      uint pivot = uint(k);
      if (p.aux == 2u) {
        uint best = 0u;
        for (uint64_t row = k; row < n; ++row) {
          uint mag = mag_i32(factor_values[
              ix(base + midx(row, k, n, n, p.storage_layout))]);
          if (mag > best) {
            best = mag;
            pivot = uint(row);
          }
        }
      }
      factor_pivot = pivot;
      aux_values[ix(batch * n + k)] = pivot;
    }
    sync_factor();
    if (factor_code == 0u && factor_pivot != uint(k)) {
      for (uint64_t col = uint64_t(lane); col < n; col += uint64_t(32)) {
        uint lhs = ix(base + midx(k, col, n, n, p.storage_layout));
        uint rhs = ix(base + midx(factor_pivot, col, n, n,
                                  p.storage_layout));
        int value = factor_values[lhs];
        factor_values[lhs] = factor_values[rhs];
        factor_values[rhs] = value;
      }
    }
    sync_factor();
    if (lane == 0u && factor_code == 0u) {
      factor_diag = factor_values[
          ix(base + midx(k, k, n, n, p.storage_layout))];
      if (factor_diag == 0) { factor_code = 1u; }
    }
    sync_factor();
    if (factor_code == 0u) {
      for (uint64_t row = k + uint64_t(1) + uint64_t(lane); row < n;
           row += uint64_t(32)) {
        uint index = ix(base + midx(row, k, n, n, p.storage_layout));
        factor_values[index] = div_q31(factor_values[index], factor_diag);
      }
    }
    sync_factor();
    uint64_t width = n - min(n, k + uint64_t(1));
    if (factor_code == 0u) {
      for (uint64_t cell = uint64_t(lane); cell < width * width;
           cell += uint64_t(32)) {
        uint64_t row = k + uint64_t(1) + cell / width;
        uint64_t col = k + uint64_t(1) + cell % width;
        uint index = ix(base + midx(row, col, n, n, p.storage_layout));
        uint multiplier = ix(base + midx(row, k, n, n, p.storage_layout));
        uint pivot = ix(base + midx(k, col, n, n, p.storage_layout));
        factor_values[index] = sub_q31(
            factor_values[index],
            mul_q31(factor_values[multiplier], factor_values[pivot]));
      }
    }
    sync_factor();
  }
  return factor_code;
}

uint factor_cholesky(uint64_t batch) {
  uint lane = gl_LocalInvocationID.x;
  uint64_t n = p.rows;
  uint64_t base = batch * n * n;
  if (lane == 0u) { factor_code = 0u; }
  for (uint64_t index = uint64_t(lane); index < n * n;
       index += uint64_t(32)) {
    factor_values[ix(base + index)] = 0;
  }
  sync_factor();
  for (uint64_t col = uint64_t(0); col < n; ++col) {
    if (lane == 0u && factor_code == 0u) {
      int sum = input_values[
          ix(base + midx(col, col, n, n, p.storage_layout))];
      for (uint64_t k = uint64_t(0); k < col; ++k) {
        int value = factor_values[
            ix(base + midx(col, k, n, n, p.storage_layout))];
        sum = sub_q31(sum, mul_q31(value, value));
      }
      if (sum <= 0) {
        factor_code = 2u;
      } else {
        factor_diag = sqrt_q31(sum);
        factor_values[ix(base + midx(col, col, n, n,
                                     p.storage_layout))] = factor_diag;
      }
    }
    sync_factor();
    if (factor_code == 0u) {
      for (uint64_t row = col + uint64_t(1) + uint64_t(lane); row < n;
           row += uint64_t(32)) {
        int sum = input_values[
            ix(base + midx(row, col, n, n, p.storage_layout))];
        for (uint64_t k = uint64_t(0); k < col; ++k) {
          sum = sub_q31(
              sum, mul_q31(
                       factor_values[ix(base + midx(row, k, n, n,
                                                     p.storage_layout))],
                       factor_values[ix(base + midx(col, k, n, n,
                                                     p.storage_layout))]));
        }
        factor_values[ix(base + midx(row, col, n, n, p.storage_layout))] =
            div_q31(sum, factor_diag);
      }
    }
    sync_factor();
  }
  return factor_code;
}

uint factor_qr(uint64_t batch) {
  uint lane = gl_LocalInvocationID.x;
  uint rows = uint(p.rows);
  uint cols = uint(p.cols);
  if (lane == 0u) {
    factor_code = (rows > 16u || cols > 16u) ? 4u : 0u;
  }
  for (uint index = lane; index < 256u; index += 32u) {
    factor_q[index] = 0;
    factor_r[index] = 0;
  }
  sync_factor();
  uint64_t input_base = batch * p.rows * p.cols;
  for (uint col = 0u; col < cols; ++col) {
    if (factor_code == 0u) {
      for (uint row = lane; row < rows; row += 32u) {
        factor_v[row] = input_values[ix(
            input_base + midx(row, col, p.rows, p.cols, p.storage_layout))];
      }
    }
    sync_factor();
    for (uint previous = 0u; previous < col; ++previous) {
      if (lane == 0u && factor_code == 0u) {
        factor_dot = 0;
        for (uint row = 0u; row < rows; ++row) {
          factor_dot = add_q31(
              factor_dot,
              mul_q31(factor_q[row * cols + previous], factor_v[row]));
        }
        factor_r[previous * cols + col] = factor_dot;
      }
      sync_factor();
      if (factor_code == 0u) {
        for (uint row = lane; row < rows; row += 32u) {
          factor_v[row] = sub_q31(
              factor_v[row],
              mul_q31(factor_dot, factor_q[row * cols + previous]));
        }
      }
      sync_factor();
    }
    if (lane == 0u && factor_code == 0u) {
      int squared = 0;
      for (uint row = 0u; row < rows; ++row) {
        squared = add_q31(squared,
                          mul_q31(factor_v[row], factor_v[row]));
      }
      factor_norm = sqrt_q31(squared);
      if (factor_norm == 0) {
        factor_code = 1u;
      } else {
        factor_r[col * cols + col] = factor_norm;
      }
    }
    sync_factor();
    if (factor_code == 0u) {
      for (uint row = lane; row < rows; row += 32u) {
        factor_q[row * cols + col] = div_q31(factor_v[row], factor_norm);
      }
    }
    sync_factor();
  }
  uint64_t output_base = batch * p.value_count;
  if (factor_code == 0u) {
    for (uint cell = lane; cell < rows * cols; cell += 32u) {
      uint row = cell / cols;
      uint col = cell % cols;
      factor_values[ix(output_base +
                       midx(row, col, p.rows, p.cols, p.storage_layout))] =
          factor_q[cell];
    }
    if (p.mode == 2u) {
      uint64_t r_base = output_base + p.rows * p.cols;
      for (uint cell = lane; cell < cols * cols; cell += 32u) {
        uint row = cell / cols;
        uint col = cell % cols;
        factor_values[ix(r_base +
                         midx(row, col, p.cols, p.cols, p.storage_layout))] =
            factor_r[cell];
      }
    }
  }
  sync_factor();
  return factor_code;
}

void main() {
  uint64_t batch = uint64_t(gl_WorkGroupID.x);
  if (batch >= p.batch_count) { return; }
  uint code = 4u;
  if (p.op == uint64_t(1)) { code = factor_lu(batch); }
  if (p.op == uint64_t(2)) { code = factor_qr(batch); }
  if (p.op == uint64_t(3)) { code = factor_cholesky(batch); }
  if (gl_LocalInvocationID.x == 0u) { status_values[ix(batch)] = code; }
}
)GLSL");
}

} // namespace

[[nodiscard]] std::string FactorSource() {
  std::uint64_t exact_bytes = 0u;
  const auto emit = [](auto &sink)
      noexcept(noexcept(EmitFactorSource(sink))) {
    return EmitFactorSource(sink);
  };
  return backend_source_recipe::bytes(emit, exact_bytes)
             ? backend_source_recipe::materialize(emit, exact_bytes)
             : std::string{};
}

bool FactorSourceBytes(std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [](backend_source_recipe::CountSink &sink) noexcept {
        return EmitFactorSource(sink);
      },
      bytes);
}

} // namespace rund::node::accel::detail
