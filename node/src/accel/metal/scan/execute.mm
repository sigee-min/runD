#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../buffer/resident/batch.hpp"
#include "local.hpp"

#include <kernel/program/compute/model.hpp>

namespace rund::node::accel::detail {

rund::AccelCheck ExecuteMetalScan(const rund::AccelDevice &pick,
                                  const rund::kernel::ScanDesc &desc,
                                  const rund::kernel::ScanPlan &plan,
                                  const rund::kernel::ComputeDomain domain,
                                  const ScanBinds &bindings) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  if (!MetalPickOwnsAdapter(pick)) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (adapter == nullptr || adapter->device == nullptr ||
      adapter->queue == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  SetMetalLastError(*adapter, "ok");
  if (!ScanShapeOk(desc, plan) || !ScanResidentShapeOk(plan, bindings)) {
    SetMetalLastError(*adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  MetalResidentBufferResult input{};
  MetalResidentBufferResult output{};
  MetalResidentBufferResult logical_count{};
  MetalResidentReq reqs[] = {
      {bindings.input, bindings.input_handle, &input},
      {bindings.output, bindings.output_handle, &output}};
  LookupMetalResidentBatch(pick, reqs, "accel_metal_resident_id_unavailable");
  if (bindings.logical_count_handle != nullptr) {
    MetalResidentReq count[] = {{bindings.logical_count,
                                 bindings.logical_count_handle,
                                 &logical_count}};
    LookupMetalResidentBatch(pick, count,
                             "accel_metal_resident_id_unavailable");
  }
  if (!input.check.ok || !output.check.ok || input.device_buffer == nullptr ||
      output.device_buffer == nullptr ||
      (bindings.logical_count_handle != nullptr &&
       (!logical_count.check.ok || logical_count.device_buffer == nullptr))) {
    const char *const reason =
        !input.check.ok ? input.check.reason
                        : (!output.check.ok ? output.check.reason
                                            : logical_count.check.reason);
    SetMetalLastError(*adapter, reason);
    return rund::AccelCheck{false, reason};
  }
  return ExecuteMetalScanBuffers(
      *adapter, desc, plan, domain, input.device_buffer.get(),
      output.device_buffer.get(), true,
      logical_count.device_buffer == nullptr
          ? nullptr
          : logical_count.device_buffer.get(),
      static_cast<rund::kernel::u32>(
          rund::kernel::ComputeCountBytes(plan.count_source) /
          sizeof(rund::kernel::u32)));
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)bindings;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}
} // namespace rund::node::accel::detail
