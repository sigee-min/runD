#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../buffer/resident/batch.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
LookupMetalScanBuffers(const rund::AccelDevice &pick,
                       const ScanBinds &bindings,
                       MetalScanEncodeResources &resources) {
  MetalResidentReq reqs[] = {
      {bindings.input, bindings.input_handle, &resources.input},
      {bindings.output, bindings.output_handle, &resources.output}};
  LookupMetalResidentBatch(pick, reqs, "accel_metal_resident_id_unavailable");
  if (bindings.logical_count_handle != nullptr) {
    MetalResidentReq count[] = {{bindings.logical_count,
                                 bindings.logical_count_handle,
                                 &resources.logical_count}};
    LookupMetalResidentBatch(pick, count,
                             "accel_metal_resident_id_unavailable");
  }
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

} // namespace
#endif
} // namespace rund::node::accel::detail
