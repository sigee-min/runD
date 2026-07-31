#pragma once

#include <accel/check.hpp>

#include "finish.hpp"
#include "scratch.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck ExecuteMetalScanBuffers(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan,
    const rund::kernel::ComputeDomain domain, void *const input_buffer,
    void *const output_buffer, const bool record_dispatches,
    void *const logical_count_buffer, const rund::kernel::u32 count_words) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  if (adapter.device == nullptr || adapter.queue == nullptr ||
      input_buffer == nullptr || output_buffer == nullptr) {
    SetMetalLastError(adapter, "accel_metal_unavailable");
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  SetMetalLastError(adapter, "ok");
  if (!ScanShapeOk(desc, plan)) {
    SetMetalLastError(adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }

  MetalScanDirectBuffers buffers{};
  rund::AccelCheck check =
      AcquireMetalScanDirectBuffers(adapter, plan, buffers);
  if (check.ok) {
    CommandRun command{};
    check = OpenCommand(adapter, command);
    if (check.ok) {
      check = EncodeMetalScanDirectBuffers(
          adapter, desc, plan, domain, input_buffer, output_buffer, buffers,
          command, logical_count_buffer, count_words);
      check = FinishMetalScanDirectBuffers(adapter, plan, record_dispatches,
                                           buffers, command, check);
    }
  }
  ReleaseMetalScanDirectBuffers(adapter, buffers);
  return check;
#else
  (void)adapter;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)input_buffer;
  (void)output_buffer;
  (void)record_dispatches;
  (void)logical_count_buffer;
  (void)count_words;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
