#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../buffer/resident/batch.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
LookupMetalSegmentedScanBuffers(const rund::AccelDevice &pick,
                                const SegmentedScanBinds &bindings,
                                MetalSegmentedScanEncodeResources &resources) {
  MetalResidentReq reqs[] = {
      {bindings.input, bindings.input_handle, &resources.input},
      {bindings.heads, bindings.heads_handle, &resources.heads},
      {bindings.output, bindings.output_handle, &resources.output}};
  LookupMetalResidentBatch(pick, reqs, "accel_metal_resident_id_unavailable");
  if (resources.input.check.ok && resources.heads.check.ok &&
      resources.output.check.ok && resources.input.device_buffer != nullptr &&
      resources.heads.device_buffer != nullptr &&
      resources.output.device_buffer != nullptr) {
    return rund::AccelCheck{true, "ok"};
  }
  const char *const reason =
      !resources.input.check.ok
          ? resources.input.check.reason
          : (!resources.heads.check.ok ? resources.heads.check.reason
                                       : resources.output.check.reason);
  return rund::AccelCheck{false, reason};
}
#endif

} // namespace rund::node::accel::detail
