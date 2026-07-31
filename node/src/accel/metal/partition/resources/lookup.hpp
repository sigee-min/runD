#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../buffer/resident/batch.hpp"
#include "scan.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
LookupMetalPartitionResidentBuffers(const rund::AccelDevice &pick,
                                    const PartitionBinds &bindings,
                                    MetalPartitionEncodeResources &raw) {
  MetalResidentReq reqs[] = {
      {bindings.flags, bindings.flags_handle, &raw.flags},
      {bindings.values, bindings.values_handle, &raw.values},
      {bindings.output, bindings.output_handle, &raw.output}};
  LookupMetalResidentBatch(pick, reqs, "accel_metal_resident_id_unavailable");
  if (raw.flags.check.ok && raw.values.check.ok && raw.output.check.ok &&
      raw.flags.device_buffer != nullptr &&
      raw.values.device_buffer != nullptr &&
      raw.output.device_buffer != nullptr) {
    return rund::AccelCheck{true, "ok"};
  }
  const char *const reason =
      !raw.flags.check.ok ? raw.flags.check.reason
                          : (!raw.values.check.ok ? raw.values.check.reason
                                                  : raw.output.check.reason);
  SetMetalLastError(*raw.adapter, reason);
  return rund::AccelCheck{false, reason};
}
#endif

} // namespace rund::node::accel::detail
