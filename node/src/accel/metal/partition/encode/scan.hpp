#pragma once

#include <accel/check.hpp>

#include "classify.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
EncodeMetalPartitionScans(MetalAdapter &adapter,
                          const MetalPartitionCommandState &state,
                          void *const command_encoder) {
  MetalPartitionEncodeResources &partition = *state.partition;
  return EncodePreparedMetalScanBuffers(
      adapter, partition.scan_desc, partition.scan_plan,
      rund::kernel::ComputeDomain::U32, partition.false_bits.buffer.get(),
      partition.false_offsets.buffer.get(), partition.false_totals.buffer.get(),
      partition.false_status.buffer.get(), command_encoder,
      partition.scan_block, partition.scan_prefix, partition.scan_offset, nullptr,
      0u, partition.false_bits.offset, partition.false_offsets.offset, 0u,
      partition.false_totals.offset);
}
#endif

} // namespace rund::node::accel::detail
