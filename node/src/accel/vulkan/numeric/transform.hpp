#pragma once

#include <kernel/program/compute/transform/stage.hpp>

#include "../kernel/source_recipe.hpp"
#include <string_view>

namespace rund::node::accel::detail {

template <typename Sink>
[[nodiscard]] bool AppendTransformProgramSource(
    Sink &sink, const std::string_view dialect)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  VulkanSourceTextSink base{sink};
  base += "#define RUND_TRANSFORM_LANES ";
  base.decimal(kernel::transform_stage::Lanes);
  base += "u\n";
  base.append(dialect);
  base.append(R"GLSL(
layout(local_size_x = RUND_TRANSFORM_LANES) in;
layout(set = 0, binding = 1, std430) readonly buffer InReal {
  RUND_TRANSFORM_SCALAR in_r[];
};
layout(set = 0, binding = 2, std430) readonly buffer InImag {
  RUND_TRANSFORM_SCALAR in_i[];
};
layout(set = 0, binding = 3, std430) buffer OutReal {
  RUND_TRANSFORM_SCALAR out_r[];
};
layout(set = 0, binding = 4, std430) buffer OutImag {
  RUND_TRANSFORM_SCALAR out_i[];
};
layout(set = 0, binding = 5, std430) readonly buffer Twiddle {
  RUND_TRANSFORM_SCALAR twiddle[];
};
layout(push_constant) uniform TransformStage {
  uint64_t span;
  uint64_t stride;
  uint64_t next_span;
  uint64_t next_stride;
  uint64_t bit_count;
} transform_stage;

shared RUND_TRANSFORM_SCALAR block_r[RUND_TRANSFORM_LANES];
shared RUND_TRANSFORM_SCALAR block_i[RUND_TRANSFORM_LANES];

void sync_transform() {
  memoryBarrierShared();
  barrier();
}

void transform_butterfly(
    RUND_TRANSFORM_SCALAR wr, RUND_TRANSFORM_SCALAR wi,
    RUND_TRANSFORM_SCALAR ur, RUND_TRANSFORM_SCALAR ui,
    RUND_TRANSFORM_SCALAR vr, RUND_TRANSFORM_SCALAR vi,
    out RUND_TRANSFORM_SCALAR lhs_r, out RUND_TRANSFORM_SCALAR lhs_i,
    out RUND_TRANSFORM_SCALAR rhs_r, out RUND_TRANSFORM_SCALAR rhs_i) {
  RUND_TRANSFORM_SCALAR tr = RUND_TRANSFORM_SUB(
      RUND_TRANSFORM_MUL(wr, vr), RUND_TRANSFORM_MUL(wi, vi));
  RUND_TRANSFORM_SCALAR ti = RUND_TRANSFORM_ADD(
      RUND_TRANSFORM_MUL(wr, vi), RUND_TRANSFORM_MUL(wi, vr));
  lhs_r = RUND_TRANSFORM_ADD(ur, tr);
  lhs_i = RUND_TRANSFORM_ADD(ui, ti);
  rhs_r = RUND_TRANSFORM_SUB(ur, tr);
  rhs_i = RUND_TRANSFORM_SUB(ui, ti);
}

void transform_normalize(
    inout RUND_TRANSFORM_SCALAR lhs_r,
    inout RUND_TRANSFORM_SCALAR lhs_i,
    inout RUND_TRANSFORM_SCALAR rhs_r,
    inout RUND_TRANSFORM_SCALAR rhs_i) {
  if (p.aux == 1u) { return; }
  uint64_t divisor = p.batch_count;
  lhs_r = RUND_TRANSFORM_DIV(lhs_r, divisor);
  lhs_i = RUND_TRANSFORM_DIV(lhs_i, divisor);
  rhs_r = RUND_TRANSFORM_DIV(rhs_r, divisor);
  rhs_i = RUND_TRANSFORM_DIV(rhs_i, divisor);
}

uint64_t transform_reverse(uint64_t value, uint bits) {
  return bits == 0u
             ? uint64_t(0)
             : uint64_t(bitfieldReverse(uint(value))) >> (32u - bits);
}

void main() {
  uint64_t index = uint64_t(gl_GlobalInvocationID.x);
  uint64_t n = p.rows;
  uint64_t span = transform_stage.span;
  if (span == uint64_t(1)) {
    uint lane = gl_LocalInvocationID.x;
    uint64_t block_start =
        uint64_t(gl_WorkGroupID.x) * uint64_t(RUND_TRANSFORM_LANES);
    uint64_t block_length =
        min(n - block_start, uint64_t(RUND_TRANSFORM_LANES));
    if (uint64_t(lane) < block_length) {
      uint target = RUND_TRANSFORM_INDEX(block_start + uint64_t(lane));
      uint source = RUND_TRANSFORM_INDEX(transform_reverse(
          block_start + uint64_t(lane), uint(transform_stage.bit_count)));
      block_r[lane] = in_r[source];
      block_i[lane] = in_i[source];
    }
    sync_transform();
    uint64_t stride = transform_stage.stride;
    for (uint64_t local_span = uint64_t(2);
         local_span <= block_length; local_span <<= uint64_t(1)) {
      uint64_t half_span = local_span >> uint64_t(1);
      uint64_t butterflies = block_length >> uint64_t(1);
      if (uint64_t(lane) < butterflies) {
        uint64_t item = uint64_t(lane);
        uint64_t phase = item & (half_span - uint64_t(1));
        uint64_t base = (item - phase) << uint64_t(1);
        uint twiddle_index = uint(phase * stride);
        RUND_TRANSFORM_SCALAR wr = twiddle[twiddle_index];
        RUND_TRANSFORM_SCALAR wi =
            twiddle[uint(n >> uint64_t(1)) + twiddle_index];
        uint lhs = uint(base + phase);
        uint rhs = uint(base + phase + half_span);
        RUND_TRANSFORM_SCALAR lhs_r;
        RUND_TRANSFORM_SCALAR lhs_i;
        RUND_TRANSFORM_SCALAR rhs_r;
        RUND_TRANSFORM_SCALAR rhs_i;
        transform_butterfly(
            wr, wi, block_r[lhs], block_i[lhs], block_r[rhs], block_i[rhs],
            lhs_r, lhs_i, rhs_r, rhs_i);
        if (local_span == n) {
          transform_normalize(lhs_r, lhs_i, rhs_r, rhs_i);
        }
        block_r[lhs] = lhs_r;
        block_i[lhs] = lhs_i;
        block_r[rhs] = rhs_r;
        block_i[rhs] = rhs_i;
      }
      stride >>= uint64_t(1);
      sync_transform();
    }
    if (uint64_t(lane) < block_length) {
      uint target = RUND_TRANSFORM_INDEX(block_start + uint64_t(lane));
      out_r[target] = block_r[lane];
      out_i[target] = block_i[lane];
    }
    return;
  }
  bool paired = transform_stage.next_span != uint64_t(0);
  uint64_t active_count = paired ? (n >> uint64_t(2)) : (n >> uint64_t(1));
  if (index >= active_count) { return; }
  uint64_t half_span = span >> 1;
  uint64_t phase = index & (half_span - uint64_t(1));
  uint64_t base = (index - phase) <<
                  (paired ? uint64_t(2) : uint64_t(1));
  uint first_index = uint(phase * transform_stage.stride);
  RUND_TRANSFORM_SCALAR first_wr = twiddle[first_index];
  RUND_TRANSFORM_SCALAR first_wi =
      twiddle[uint(n >> uint64_t(1)) + first_index];
  uint a = RUND_TRANSFORM_INDEX(base + phase);
  uint b = RUND_TRANSFORM_INDEX(base + phase + half_span);
  RUND_TRANSFORM_SCALAR first_a_r;
  RUND_TRANSFORM_SCALAR first_a_i;
  RUND_TRANSFORM_SCALAR first_b_r;
  RUND_TRANSFORM_SCALAR first_b_i;
  transform_butterfly(
      first_wr, first_wi, out_r[a], out_i[a], out_r[b], out_i[b],
      first_a_r, first_a_i, first_b_r, first_b_i);
  if (!paired) {
    if (span == n) {
      transform_normalize(first_a_r, first_a_i, first_b_r, first_b_i);
    }
    out_r[a] = first_a_r;
    out_i[a] = first_a_i;
    out_r[b] = first_b_r;
    out_i[b] = first_b_i;
    return;
  }

  uint c = RUND_TRANSFORM_INDEX(base + phase + span);
  uint d = RUND_TRANSFORM_INDEX(base + phase + span + half_span);
  RUND_TRANSFORM_SCALAR first_c_r;
  RUND_TRANSFORM_SCALAR first_c_i;
  RUND_TRANSFORM_SCALAR first_d_r;
  RUND_TRANSFORM_SCALAR first_d_i;
  transform_butterfly(
      first_wr, first_wi, out_r[c], out_i[c], out_r[d], out_i[d],
      first_c_r, first_c_i, first_d_r, first_d_i);
  uint second_a_index = uint(phase * transform_stage.next_stride);
  uint second_b_index =
      uint((phase + half_span) * transform_stage.next_stride);
  RUND_TRANSFORM_SCALAR second_a_r;
  RUND_TRANSFORM_SCALAR second_a_i;
  RUND_TRANSFORM_SCALAR second_c_r;
  RUND_TRANSFORM_SCALAR second_c_i;
  RUND_TRANSFORM_SCALAR second_b_r;
  RUND_TRANSFORM_SCALAR second_b_i;
  RUND_TRANSFORM_SCALAR second_d_r;
  RUND_TRANSFORM_SCALAR second_d_i;
  transform_butterfly(
      twiddle[second_a_index],
      twiddle[uint(n >> uint64_t(1)) + second_a_index],
      first_a_r, first_a_i, first_c_r, first_c_i,
      second_a_r, second_a_i, second_c_r, second_c_i);
  transform_butterfly(
      twiddle[second_b_index],
      twiddle[uint(n >> uint64_t(1)) + second_b_index],
      first_b_r, first_b_i, first_d_r, first_d_i,
      second_b_r, second_b_i, second_d_r, second_d_i);
  if (transform_stage.next_span == n) {
    transform_normalize(second_a_r, second_a_i, second_c_r, second_c_i);
    transform_normalize(second_b_r, second_b_i, second_d_r, second_d_i);
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
)GLSL");
  return base.ok();
}

} // namespace rund::node::accel::detail
