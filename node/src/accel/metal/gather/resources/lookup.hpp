#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../buffer/resident/batch.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
LookupMetalGatherResidentBuffers(const rund::AccelDevice &pick,
                                 const GatherBinds &bindings,
                                 MetalGatherEncodeResources &resources) {
  MetalResidentReq reqs[] = {
      {bindings.values, bindings.values_handle, &resources.values},
      {bindings.indices, bindings.indices_handle, &resources.indices},
      {bindings.output, bindings.output_handle, &resources.output},
      {bindings.logical_count, bindings.logical_count_handle,
       &resources.logical_count}};
  const std::size_t count = bindings.logical_count_handle == nullptr ? 3u : 4u;
  LookupMetalResidentBatch(pick, reqs, count,
                           "accel_metal_resident_id_unavailable");
  if (resources.values.check.ok && resources.indices.check.ok &&
      resources.output.check.ok && resources.values.device_buffer != nullptr &&
      resources.indices.device_buffer != nullptr &&
      resources.output.device_buffer != nullptr &&
      (bindings.logical_count_handle == nullptr ||
       (resources.logical_count.check.ok &&
        resources.logical_count.device_buffer != nullptr))) {
    return rund::AccelCheck{true, "ok"};
  }
  const char *const reason =
      !resources.values.check.ok
          ? resources.values.check.reason
          : (!resources.indices.check.ok ? resources.indices.check.reason
                                         : (!resources.output.check.ok
                                                ? resources.output.check.reason
                                                : resources.logical_count.check.reason));
  return rund::AccelCheck{false, reason};
}
#endif

} // namespace rund::node::accel::detail
