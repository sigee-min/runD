#pragma once

#include <string_view>

namespace rund::node::accel::detail::source::program {

inline constexpr std::string_view Transform = R"MSL(inline void RUND_KERNEL(rund_transform_butterfly_)(
    constant NumericParams& p, RUND_SCALAR wr, RUND_SCALAR wi,
    RUND_SCALAR ur, RUND_SCALAR ui,
    RUND_SCALAR vr, RUND_SCALAR vi,
    thread RUND_SCALAR& lhs_r, thread RUND_SCALAR& lhs_i,
    thread RUND_SCALAR& rhs_r, thread RUND_SCALAR& rhs_i) {
  RUND_SCALAR tr = RUND_SUB(RUND_MUL(wr, vr), RUND_MUL(wi, vi));
  RUND_SCALAR ti = RUND_ADD(RUND_MUL(wr, vi), RUND_MUL(wi, vr));
  lhs_r = RUND_ADD(ur, tr);
  lhs_i = RUND_ADD(ui, ti);
  rhs_r = RUND_SUB(ur, tr);
  rhs_i = RUND_SUB(ui, ti);
}

inline void RUND_KERNEL(rund_transform_normalize_)(
    constant NumericParams& p, thread RUND_SCALAR& lhs_r,
    thread RUND_SCALAR& lhs_i,
    thread RUND_SCALAR& rhs_r, thread RUND_SCALAR& rhs_i) {
  if (p.aux == 1u) { return; }
  RUND_SCALAR divisor = RUND_FROM_ULONG(p.batch_count);
  lhs_r /= divisor;
  lhs_i /= divisor;
  rhs_r /= divisor;
  rhs_i /= divisor;
}

inline ulong RUND_KERNEL(rund_transform_reverse_)(ulong value, uint bits) {
  return bits == 0u ? 0ul : reverse_bits(value) >> (64u - bits);
}

kernel void RUND_KERNEL(rund_numeric_transform_)(
    device const RUND_SCALAR* in_r [[buffer(0)]],
    device const RUND_SCALAR* in_i [[buffer(1)]],
    device RUND_SCALAR* out_r [[buffer(2)]],
    device RUND_SCALAR* out_i [[buffer(3)]],
    device const RUND_SCALAR* twiddle [[buffer(4)]],
    constant NumericParams& p [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  threadgroup RUND_SCALAR block_r[RUND_TRANSFORM_LANES];
  threadgroup RUND_SCALAR block_i[RUND_TRANSFORM_LANES];
  ulong index = ulong(gid.x);
  ulong n = p.rows;
  ulong span = p.inner;
  if (span == 1ul) {
    ulong block_start = ulong(group.x) * RUND_TRANSFORM_LANES;
    ulong block_length = min(n - block_start, ulong(RUND_TRANSFORM_LANES));
    if (ulong(lane) < block_length) {
      ulong target = block_start + ulong(lane);
      ulong source = RUND_KERNEL(rund_transform_reverse_)(
          target, uint(p.max_iterations));
      block_r[lane] = in_r[source];
      block_i[lane] = in_i[source];
    }
    rund_numeric_sync();
    ulong stride = p.cols;
    for (ulong local_span = 2ul; local_span <= block_length;
         local_span <<= 1ul) {
      ulong half_span = local_span >> 1ul;
      ulong butterflies = block_length >> 1ul;
      if (ulong(lane) < butterflies) {
        ulong item = ulong(lane);
        ulong phase = item & (half_span - 1ul);
        ulong base = (item - phase) << 1ul;
        ulong twiddle_index = phase * stride;
        RUND_SCALAR wr = twiddle[twiddle_index];
        RUND_SCALAR wi = twiddle[(n >> 1ul) + twiddle_index];
        ulong lhs = base + phase;
        ulong rhs = lhs + half_span;
        RUND_SCALAR lhs_r;
        RUND_SCALAR lhs_i;
        RUND_SCALAR rhs_r;
        RUND_SCALAR rhs_i;
        RUND_KERNEL(rund_transform_butterfly_)(
            p, wr, wi, block_r[lhs], block_i[lhs], block_r[rhs],
            block_i[rhs], lhs_r, lhs_i, rhs_r, rhs_i);
        if (local_span == n) {
          RUND_KERNEL(rund_transform_normalize_)(
              p, lhs_r, lhs_i, rhs_r, rhs_i);
        }
        block_r[lhs] = lhs_r;
        block_i[lhs] = lhs_i;
        block_r[rhs] = rhs_r;
        block_i[rhs] = rhs_i;
      }
      stride >>= 1ul;
      rund_numeric_sync();
    }
    if (ulong(lane) < block_length) {
      out_r[block_start + ulong(lane)] = block_r[lane];
      out_i[block_start + ulong(lane)] = block_i[lane];
    }
    return;
  }
  bool paired = p.rhs_cols != 0ul;
  ulong active_count = paired ? (n >> 2ul) : (n >> 1ul);
  if (index >= active_count) { return; }
  ulong half_span = span >> 1ul;
  ulong phase = index & (half_span - 1ul);
  ulong base = (index - phase) << (paired ? 2ul : 1ul);
  ulong first_index = phase * p.cols;
  RUND_SCALAR first_wr = twiddle[first_index];
  RUND_SCALAR first_wi = twiddle[(n >> 1ul) + first_index];
  ulong a = base + phase;
  ulong b = a + half_span;
  RUND_SCALAR first_a_r;
  RUND_SCALAR first_a_i;
  RUND_SCALAR first_b_r;
  RUND_SCALAR first_b_i;
  RUND_KERNEL(rund_transform_butterfly_)(
      p, first_wr, first_wi, out_r[a], out_i[a], out_r[b], out_i[b],
      first_a_r, first_a_i, first_b_r, first_b_i);
  if (!paired) {
    if (span == n) {
      RUND_KERNEL(rund_transform_normalize_)(
          p, first_a_r, first_a_i, first_b_r, first_b_i);
    }
    out_r[a] = first_a_r;
    out_i[a] = first_a_i;
    out_r[b] = first_b_r;
    out_i[b] = first_b_i;
    return;
  }

  ulong c = a + span;
  ulong d = c + half_span;
  RUND_SCALAR first_c_r;
  RUND_SCALAR first_c_i;
  RUND_SCALAR first_d_r;
  RUND_SCALAR first_d_i;
  RUND_KERNEL(rund_transform_butterfly_)(
      p, first_wr, first_wi, out_r[c], out_i[c], out_r[d], out_i[d],
      first_c_r, first_c_i, first_d_r, first_d_i);
  ulong second_a_index = phase * p.value_count;
  ulong second_b_index = (phase + half_span) * p.value_count;
  RUND_SCALAR second_a_r;
  RUND_SCALAR second_a_i;
  RUND_SCALAR second_c_r;
  RUND_SCALAR second_c_i;
  RUND_SCALAR second_b_r;
  RUND_SCALAR second_b_i;
  RUND_SCALAR second_d_r;
  RUND_SCALAR second_d_i;
  RUND_KERNEL(rund_transform_butterfly_)(
      p, twiddle[second_a_index], twiddle[(n >> 1ul) + second_a_index],
      first_a_r, first_a_i, first_c_r, first_c_i,
      second_a_r, second_a_i, second_c_r, second_c_i);
  RUND_KERNEL(rund_transform_butterfly_)(
      p, twiddle[second_b_index], twiddle[(n >> 1ul) + second_b_index],
      first_b_r, first_b_i, first_d_r, first_d_i,
      second_b_r, second_b_i, second_d_r, second_d_i);
  if (p.rhs_cols == n) {
    RUND_KERNEL(rund_transform_normalize_)(
        p, second_a_r, second_a_i, second_c_r, second_c_i);
    RUND_KERNEL(rund_transform_normalize_)(
        p, second_b_r, second_b_i, second_d_r, second_d_i);
  }
  out_r[a] = second_a_r;
  out_i[a] = second_a_i;
  out_r[b] = second_b_r;
  out_i[b] = second_b_i;
  out_r[c] = second_c_r;
  out_i[c] = second_c_i;
  out_r[d] = second_d_r;
  out_i[d] = second_d_i;
}

)MSL";

} // namespace rund::node::accel::detail::source::program
