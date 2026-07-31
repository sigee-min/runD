#pragma once

#include <accel/device.hpp>

#include "../../buffer/resident/batch.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct ScatterBufferLookup {
  VulkanResidentBufferResult values{};
  VulkanResidentBufferResult indices{};
  VulkanResidentBufferResult output{};
};

[[nodiscard]] ScatterBufferLookup
LookupScatterBuffers(const rund::AccelDevice &pick,
                     const ScatterBinds &bindings) {
  ScatterBufferLookup lookup{};
  VulkanResidentReq reqs[] = {
      {bindings.values, bindings.values_handle, &lookup.values},
      {bindings.indices, bindings.indices_handle, &lookup.indices},
      {bindings.output, bindings.output_handle, &lookup.output}};
  LookupVulkanResidentBatch(pick, reqs, "compute_resident_id_invalid");
  return lookup;
}

[[nodiscard]] const char *
ScatterLookupReason(const ScatterBufferLookup &lookup) {
  return !lookup.values.check.ok
             ? lookup.values.check.reason
             : (!lookup.indices.check.ok ? lookup.indices.check.reason
                                         : lookup.output.check.reason);
}

[[nodiscard]] bool ScatterLookupOk(const ScatterBufferLookup &lookup) {
  return lookup.values.check.ok && lookup.indices.check.ok &&
         lookup.output.check.ok && lookup.values.device_buffer != nullptr &&
         lookup.indices.device_buffer != nullptr &&
         lookup.output.device_buffer != nullptr;
}

} // namespace
#endif

} // namespace rund::node::accel::detail
