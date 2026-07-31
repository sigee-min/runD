#pragma once

#include <accel/check.hpp>

#include "offset.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck EncodeMetalScanU32FlagBuffers(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, void *const flags_buffer,
    const std::uint64_t flags_offset, void *const output_buffer,
    const std::uint64_t output_offset, void *const totals_buffer,
    const std::uint64_t totals_offset, void *const status_buffer,
    void *const command_encoder,
    const bool materialize_offsets) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  MetalFlagScanEncodeState state{};
  const rund::AccelCheck prepared = PrepareMetalFlagScanEncodeState(
      adapter, desc, plan, flags_buffer, flags_offset, output_buffer,
      output_offset, totals_buffer, totals_offset, status_buffer,
      command_encoder, state);
  if (!prepared.ok) {
    return prepared;
  }
  EncodeMetalFlagScanBlock(state);
  if (plan.pass_count == 2u) {
    EncodeMetalFlagScanPrefix(state);
    if (materialize_offsets) {
      EncodeMetalFlagScanOffset(state);
    }
  }
  SetMetalLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)desc;
  (void)plan;
  (void)flags_buffer;
  (void)flags_offset;
  (void)output_buffer;
  (void)output_offset;
  (void)totals_buffer;
  (void)totals_offset;
  (void)status_buffer;
  (void)command_encoder;
  (void)materialize_offsets;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
