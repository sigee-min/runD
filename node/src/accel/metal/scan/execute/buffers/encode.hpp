#pragma once

#include <accel/check.hpp>

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck EncodeMetalScanDirectBuffers(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan,
    const rund::kernel::ComputeDomain domain, void *const input_buffer,
    void *const output_buffer, const MetalScanDirectBuffers &buffers,
    const CommandRun &command, void *const logical_count_buffer,
    const rund::kernel::u32 count_words) {
  return EncodeMetalScanBuffers(
      adapter, desc, plan, domain, input_buffer, output_buffer,
      buffers.totals.buffer.get(), buffers.status.buffer.get(),
      (__bridge void *)command.encoder, logical_count_buffer, count_words);
}
#endif

} // namespace rund::node::accel::detail
