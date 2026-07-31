#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../buffer/resident/batch.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
LookupMetalScatterResidentBuffers(const rund::AccelDevice &pick,
                                  const ScatterBinds &bindings,
                                  MetalScatterEncodeResources &resources) {
  MetalResidentReq reqs[] = {
      {bindings.values, bindings.values_handle, &resources.values},
      {bindings.indices, bindings.indices_handle, &resources.indices},
      {bindings.output, bindings.output_handle, &resources.output}};
  LookupMetalResidentBatch(pick, reqs, "accel_metal_resident_id_unavailable");
  if (resources.values.check.ok && resources.indices.check.ok &&
      resources.output.check.ok && resources.values.device_buffer != nullptr &&
      resources.indices.device_buffer != nullptr &&
      resources.output.device_buffer != nullptr) {
    return rund::AccelCheck{true, "ok"};
  }
  const char *const reason =
      !resources.values.check.ok
          ? resources.values.check.reason
          : (!resources.indices.check.ok ? resources.indices.check.reason
                                         : resources.output.check.reason);
  return rund::AccelCheck{false, reason};
}
#endif

} // namespace rund::node::accel::detail
