#pragma once

#include <string_view>

namespace rund::node::accel::detail::source::program::solve {

inline constexpr std::string_view Kernel =
    R"MSL(kernel void RUND_KERNEL(rund_numeric_solve_)(
    device const RUND_SCALAR* primary [[buffer(0)]],
    device const uint* aux [[buffer(1)]],
    device const RUND_SCALAR* rhs [[buffer(2)]],
    device RUND_SCALAR* output [[buffer(3)]],
    device uint* status [[buffer(4)]],
    constant NumericParams& p [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 group_size [[threads_per_threadgroup]]) {
  ulong batch = ulong(group.x);
  if (batch >= p.batch_count) { return; }
  threadgroup RUND_SCALAR a[256];
  threadgroup RUND_SCALAR x[256];
  threadgroup RUND_SCALAR multipliers[16];
  threadgroup uint control[4];
  uint lanes = group_size.x;
  if (p.mode == 1u) {
    if (p.op == 3ul) {
      rund_solve_direct_cholesky(primary, rhs, output, p, batch, lane, lanes,
                                 a, x, control);
    } else {
      rund_solve_direct_lu(primary, rhs, output, p, batch, lane, lanes, a, x,
                           multipliers, control);
    }
  } else {
    ulong stride = p.op == 2ul ? p.rows * p.rows * 2ul
                               : p.rows * p.rows;
    device const RUND_SCALAR* factor = primary + batch * stride;
    if (p.op == 1ul) {
      rund_solve_factor_lu(factor, aux, rhs, output, p, batch, lane, lanes,
                           control);
    } else if (p.op == 3ul) {
      rund_solve_factor_cholesky(factor, rhs, output, p, batch, lane, lanes,
                                 control);
    } else {
      rund_solve_factor_qr(factor, rhs, output, p, batch, lane, lanes,
                           control);
    }
  }
  if (lane == 0u) { status[batch] = control[0]; }
}

)MSL";

} // namespace rund::node::accel::detail::source::program::solve
