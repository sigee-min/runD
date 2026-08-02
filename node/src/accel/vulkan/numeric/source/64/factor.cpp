#include "../../source.hpp"

namespace rund::node::accel::detail {
namespace {

template <typename Sink>
[[nodiscard]] bool EmitFactorSource64(Sink &sink)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  return EmitNumericBaseSource(sink, true) && sink.append(R"GLSL(
layout(local_size_x = 32) in;
layout(set = 0, binding = 1, std430) readonly buffer Input { int64_t input_values[]; };
layout(set = 0, binding = 2, std430) buffer Factor { int64_t factor_values[]; };
layout(set = 0, binding = 3, std430) buffer Aux { uint aux_values[]; };
layout(set = 0, binding = 4, std430) buffer Status { uint status_values[]; };

shared int64_t factor_q64[256];
shared int64_t factor_r64[256];
shared int64_t factor_v64[16];
shared int64_t factor_diag64;
shared int64_t factor_dot64;
shared int64_t factor_norm64;
shared uint factor_code64;
shared uint factor_pivot64;

void sync_factor64() {
  memoryBarrierShared();
  memoryBarrierBuffer();
  barrier();
}

uint factor_lu64(uint64_t batch) {
  uint lane = gl_LocalInvocationID.x;
  uint64_t n = p.rows;
  uint64_t base = batch * n * n;
  if (lane == 0u) { factor_code64 = 0u; }
  sync_factor64();
  for (uint64_t i = uint64_t(lane); i < n * n; i += uint64_t(32)) {
    factor_values[ix64(base + i)] = input_values[ix64(base + i)];
  }
  sync_factor64();
  for (uint64_t k = uint64_t(0); k < n; ++k) {
    if (lane == 0u && factor_code64 == 0u) {
      uint pivot = uint(k);
      if (p.aux == 2u) {
        uint64_t best = uint64_t(0);
        for (uint64_t row = k; row < n; ++row) {
          uint64_t mag = RundAbsMagnitude64(as_u64(factor_values[
              ix64(base + midx64(row, k, n, n, p.storage_layout))]));
          if (mag > best) {
            best = mag;
            pivot = uint(row);
          }
        }
      }
      factor_pivot64 = pivot;
      aux_values[ix64(batch * n + k)] = pivot;
    }
    sync_factor64();
    if (factor_code64 == 0u && factor_pivot64 != uint(k)) {
      for (uint64_t col = uint64_t(lane); col < n; col += uint64_t(32)) {
        uint lhs = ix64(base + midx64(k, col, n, n, p.storage_layout));
        uint rhs = ix64(base + midx64(factor_pivot64, col, n, n,
                                      p.storage_layout));
        int64_t value = factor_values[lhs];
        factor_values[lhs] = factor_values[rhs];
        factor_values[rhs] = value;
      }
    }
    sync_factor64();
    if (lane == 0u && factor_code64 == 0u) {
      factor_diag64 = factor_values[
          ix64(base + midx64(k, k, n, n, p.storage_layout))];
      if (factor_diag64 == int64_t(0)) { factor_code64 = 1u; }
    }
    sync_factor64();
    if (factor_code64 == 0u) {
      for (uint64_t row = k + uint64_t(1) + uint64_t(lane); row < n;
           row += uint64_t(32)) {
        uint index = ix64(base + midx64(row, k, n, n, p.storage_layout));
        factor_values[index] = div_q63(factor_values[index], factor_diag64);
      }
    }
    sync_factor64();
    uint64_t width = n - min(n, k + uint64_t(1));
    if (factor_code64 == 0u) {
      for (uint64_t cell = uint64_t(lane); cell < width * width;
           cell += uint64_t(32)) {
        uint64_t row = k + uint64_t(1) + cell / width;
        uint64_t col = k + uint64_t(1) + cell % width;
        uint index = ix64(base + midx64(row, col, n, n, p.storage_layout));
        uint multiplier =
            ix64(base + midx64(row, k, n, n, p.storage_layout));
        uint pivot = ix64(base + midx64(k, col, n, n, p.storage_layout));
        factor_values[index] = sub_q63(
            factor_values[index],
            mul_q63(factor_values[multiplier], factor_values[pivot]));
      }
    }
    sync_factor64();
  }
  return factor_code64;
}

uint factor_cholesky64(uint64_t batch) {
  uint lane = gl_LocalInvocationID.x;
  uint64_t n = p.rows;
  uint64_t base = batch * n * n;
  if (lane == 0u) { factor_code64 = 0u; }
  for (uint64_t index = uint64_t(lane); index < n * n;
       index += uint64_t(32)) {
    factor_values[ix64(base + index)] = int64_t(0);
  }
  sync_factor64();
  for (uint64_t col = uint64_t(0); col < n; ++col) {
    if (lane == 0u && factor_code64 == 0u) {
      int64_t sum = input_values[
          ix64(base + midx64(col, col, n, n, p.storage_layout))];
      for (uint64_t k = uint64_t(0); k < col; ++k) {
        int64_t value = factor_values[
            ix64(base + midx64(col, k, n, n, p.storage_layout))];
        sum = sub_q63(sum, mul_q63(value, value));
      }
      if (sum <= int64_t(0)) {
        factor_code64 = 2u;
      } else {
        factor_diag64 = sqrt_q63(sum);
        factor_values[ix64(base + midx64(col, col, n, n,
                                         p.storage_layout))] = factor_diag64;
      }
    }
    sync_factor64();
    if (factor_code64 == 0u) {
      for (uint64_t row = col + uint64_t(1) + uint64_t(lane); row < n;
           row += uint64_t(32)) {
        int64_t sum = input_values[
            ix64(base + midx64(row, col, n, n, p.storage_layout))];
        for (uint64_t k = uint64_t(0); k < col; ++k) {
          sum = sub_q63(
              sum, mul_q63(
                       factor_values[ix64(base + midx64(row, k, n, n,
                                                         p.storage_layout))],
                       factor_values[ix64(base + midx64(col, k, n, n,
                                                         p.storage_layout))]));
        }
        factor_values[
            ix64(base + midx64(row, col, n, n, p.storage_layout))] =
            div_q63(sum, factor_diag64);
      }
    }
    sync_factor64();
  }
  return factor_code64;
}

uint factor_qr64(uint64_t batch) {
  uint lane = gl_LocalInvocationID.x;
  uint rows = uint(p.rows);
  uint cols = uint(p.cols);
  if (lane == 0u) {
    factor_code64 = (rows > 16u || cols > 16u) ? 4u : 0u;
  }
  for (uint index = lane; index < 256u; index += 32u) {
    factor_q64[index] = int64_t(0);
    factor_r64[index] = int64_t(0);
  }
  sync_factor64();
  uint64_t in_base = batch * p.rows * p.cols;
  uint64_t out_base = batch * p.value_count;
  for (uint col = 0u; col < cols; ++col) {
    if (factor_code64 == 0u) {
      for (uint row = lane; row < rows; row += 32u) {
        factor_v64[row] = input_values[ix64(
            in_base + midx64(row, col, p.rows, p.cols, p.storage_layout))];
      }
    }
    sync_factor64();
    for (uint previous = 0u; previous < col; ++previous) {
      if (lane == 0u && factor_code64 == 0u) {
        factor_dot64 = int64_t(0);
        for (uint row = 0u; row < rows; ++row) {
          factor_dot64 = add_q63(
              factor_dot64,
              mul_q63(factor_q64[row * cols + previous], factor_v64[row]));
        }
        factor_r64[previous * cols + col] = factor_dot64;
      }
      sync_factor64();
      if (factor_code64 == 0u) {
        for (uint row = lane; row < rows; row += 32u) {
          factor_v64[row] = sub_q63(
              factor_v64[row],
              mul_q63(factor_dot64,
                      factor_q64[row * cols + previous]));
        }
      }
      sync_factor64();
    }
    if (lane == 0u && factor_code64 == 0u) {
      int64_t squared = int64_t(0);
      for (uint row = 0u; row < rows; ++row) {
        squared = add_q63(
            squared, mul_q63(factor_v64[row], factor_v64[row]));
      }
      factor_norm64 = sqrt_q63(squared);
      if (factor_norm64 == int64_t(0)) {
        factor_code64 = 1u;
      } else {
        factor_r64[col * cols + col] = factor_norm64;
      }
    }
    sync_factor64();
    if (factor_code64 == 0u) {
      for (uint row = lane; row < rows; row += 32u) {
        factor_q64[row * cols + col] =
            div_q63(factor_v64[row], factor_norm64);
      }
    }
    sync_factor64();
  }
  if (factor_code64 == 0u) {
    for (uint cell = lane; cell < rows * cols; cell += 32u) {
      uint row = cell / cols;
      uint col = cell % cols;
      factor_values[ix64(
          out_base + midx64(row, col, p.rows, p.cols, p.storage_layout))] =
          factor_q64[cell];
    }
    if (p.mode == 2u) {
      uint64_t r_base = out_base + p.rows * p.cols;
      for (uint cell = lane; cell < cols * cols; cell += 32u) {
        uint row = cell / cols;
        uint col = cell % cols;
        factor_values[ix64(
            r_base + midx64(row, col, p.cols, p.cols, p.storage_layout))] =
            factor_r64[cell];
      }
    }
  }
  sync_factor64();
  return factor_code64;
}

void main() {
  uint64_t batch = uint64_t(gl_WorkGroupID.x);
  if (batch >= p.batch_count) { return; }
  uint status = p.op == uint64_t(1) ? factor_lu64(batch)
              : (p.op == uint64_t(3) ? factor_cholesky64(batch)
                                      : factor_qr64(batch));
  if (gl_LocalInvocationID.x == 0u) { status_values[ix64(batch)] = status; }
}
)GLSL");
}

} // namespace

[[nodiscard]] std::string FactorSource64() {
  std::uint64_t exact_bytes = 0u;
  const auto emit = [](auto &sink)
      noexcept(noexcept(EmitFactorSource64(sink))) {
    return EmitFactorSource64(sink);
  };
  return backend_source_recipe::bytes(emit, exact_bytes)
             ? backend_source_recipe::materialize(emit, exact_bytes)
             : std::string{};
}

bool FactorSource64Bytes(std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [](backend_source_recipe::CountSink &sink) noexcept {
        return EmitFactorSource64(sink);
      },
      bytes);
}

} // namespace rund::node::accel::detail
