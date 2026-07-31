#pragma once

#include <accel/device.hpp>

#include "../../buffer/resident/batch.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
inline void LookupMetalReduceCount(const rund::AccelDevice &pick,
                                   const ReduceBinds &bindings,
                                   MetalReduceEncodeResources &resources) {
  if (bindings.logical_count_handle == nullptr) {
    return;
  }
  MetalResidentReq count[] = {{bindings.logical_count,
                               bindings.logical_count_handle,
                               &resources.logical_count}};
  LookupMetalResidentBatch(pick, count, "accel_metal_resident_id_unavailable");
}
#endif

} // namespace rund::node::accel::detail
