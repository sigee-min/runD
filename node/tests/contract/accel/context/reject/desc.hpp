#pragma once

#include <accel/context/buffer/descriptor.hpp>

#include <node/accel/context.hpp>

#include "local.hpp"

namespace node_accel_contract::context::reject {

[[nodiscard]] inline bool RejectsForgedTypedDescriptors(const State &state) {
  rund::AccelBufferDesc overflowing = state.typed_desc;
  overflowing.scalar_width_bytes = std::numeric_limits<std::uint64_t>::max();
  overflowing.count = 2u;
  if (!CheckReason(rund::node::accel::OpenAccelBuffer(state.context,
                                                      state.buffer, overflowing)
                       .check,
                   "accel_context_buffer_overflow")) {
    return false;
  }

  rund::AccelBufferDesc too_wide = state.typed_desc;
  too_wide.count = 17u;
  if (!CheckReason(rund::node::accel::OpenAccelBuffer(state.context,
                                                      state.buffer, too_wide)
                       .check,
                   "accel_context_buffer_overflow")) {
    return false;
  }

  return true;
}

} // namespace node_accel_contract::context::reject
