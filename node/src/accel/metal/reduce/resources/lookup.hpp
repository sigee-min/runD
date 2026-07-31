#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../buffer/resident/batch.hpp"
#include "count.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
LookupMetalReduceResidentBuffers(const rund::AccelDevice &pick,
                                 const ReduceBinds &bindings,
                                 MetalReduceEncodeResources &resources) {
  MetalResidentReq reqs[] = {
      {bindings.input, bindings.input_handle, &resources.input},
      {bindings.output, bindings.output_handle, &resources.output}};
  LookupMetalResidentBatch(pick, reqs, "accel_metal_resident_id_unavailable");
  LookupMetalReduceCount(pick, bindings, resources);
  if (resources.input.check.ok && resources.output.check.ok &&
      resources.input.device_buffer != nullptr &&
      resources.output.device_buffer != nullptr &&
      (bindings.logical_count_handle == nullptr ||
       (resources.logical_count.check.ok &&
        resources.logical_count.device_buffer != nullptr))) {
    return rund::AccelCheck{true, "ok"};
  }
  const char *const reason =
      !resources.input.check.ok
          ? resources.input.check.reason
          : (!resources.output.check.ok ? resources.output.check.reason
                                        : resources.logical_count.check.reason);
  return rund::AccelCheck{false, reason};
}
#endif
} // namespace rund::node::accel::detail
