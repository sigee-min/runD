#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] bool PrepareStagedOutputBuffer(
    MetalAdapter &adapter, const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow &window,
    ScopedMetalBuffers &scoped, MetalRuntimeBuffer *&output_buffer,
    std::size_t &output_size) {
  output_buffer = nullptr;
  output_size = 0u;
  if (!rund::kernel::checked::mul(window.tile_count,
                                  plan.output_bytes_per_tile)) {
    return false;
  }
  const rund::kernel::u64 output_bytes =
      window.tile_count * plan.output_bytes_per_tile;
  if (!ToSize(output_bytes, output_size)) {
    return false;
  }
  MetalRuntimeBuffer &staged_output = scoped.add(
      AcquireMetalBuffer(adapter, output_bytes, MetalBufferUsage::Output));
  if (staged_output.buffer == nullptr) {
    return false;
  }
  output_buffer = &staged_output;
  return true;
}

} // namespace
#endif

} // namespace rund::node::accel::detail
