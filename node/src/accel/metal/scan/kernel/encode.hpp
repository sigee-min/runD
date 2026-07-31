#pragma once

#include <accel/check.hpp>

#include "local.hpp"

#include <kernel/program/compute/model.hpp>

namespace rund::node::accel::detail {

rund::AccelCheck EncodeMetalScan(MetalAdapter &adapter,
                                 const std::shared_ptr<void> &resources,
                                 void *command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const scan = static_cast<MetalScanEncodeResources *>(resources.get());
  if (scan == nullptr || scan->adapter != &adapter ||
      scan->input.device_buffer == nullptr ||
      scan->output.device_buffer == nullptr || scan->totals.buffer == nullptr ||
      scan->status.buffer == nullptr) {
    SetMetalLastError(adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  return EncodePreparedMetalScanBuffers(
      adapter, scan->desc, scan->plan, scan->domain,
      scan->input.device_buffer.get(), scan->output.device_buffer.get(),
      scan->totals.buffer.get(), scan->status.buffer.get(), command_encoder,
      scan->block, scan->prefix, scan->offset,
      scan->logical_count.device_buffer == nullptr
          ? nullptr
          : scan->logical_count.device_buffer.get(),
      rund::kernel::ComputeCountBytes(scan->plan.count_source) /
      sizeof(rund::kernel::u32),
      scan->input.ref.offset_bytes, scan->output.ref.offset_bytes,
      scan->logical_count.ref.offset_bytes, scan->totals.offset);
#else
  (void)adapter;
  (void)resources;
  (void)command_encoder;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
