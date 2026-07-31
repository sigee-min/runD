#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../buffer/resident/batch.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
LookupMetalCompactBuffers(const rund::AccelDevice &pick,
                          const CompactBinds &bindings,
                          MetalCompactEncodeResources &resources) {
  MetalResidentReq reqs[] = {
      {bindings.flags, bindings.flags_handle, &resources.flags},
      {bindings.output, bindings.output_handle, &resources.output}};
  LookupMetalResidentBatch(pick, reqs, "accel_metal_resident_id_unavailable");
  if (resources.flags.check.ok && resources.output.check.ok &&
      resources.flags.device_buffer != nullptr &&
      resources.output.device_buffer != nullptr) {
    return rund::AccelCheck{true, "ok"};
  }
  const char *const reason = !resources.flags.check.ok
                                 ? resources.flags.check.reason
                                 : resources.output.check.reason;
  return rund::AccelCheck{false, reason};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
