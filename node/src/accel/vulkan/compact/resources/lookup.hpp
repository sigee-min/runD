#pragma once

#include <accel/device.hpp>

#include "../../buffer/resident/batch.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct CompactBufferLookup {
  VulkanResidentBufferResult flags{};
  VulkanResidentBufferResult output{};
};

[[nodiscard]] CompactBufferLookup
LookupCompactBuffers(const rund::AccelDevice &pick,
                     const CompactBinds &bindings) {
  CompactBufferLookup lookup{};
  VulkanResidentReq reqs[] = {
      {bindings.flags, bindings.flags_handle, &lookup.flags},
      {bindings.output, bindings.output_handle, &lookup.output}};
  LookupVulkanResidentBatch(pick, reqs, "compute_resident_id_invalid");
  return lookup;
}

[[nodiscard]] const char *
CompactLookupReason(const CompactBufferLookup &lookup) {
  return !lookup.flags.check.ok ? lookup.flags.check.reason
                                : lookup.output.check.reason;
}

[[nodiscard]] bool CompactLookupOk(const CompactBufferLookup &lookup) {
  return lookup.flags.check.ok && lookup.output.check.ok &&
         lookup.flags.device_buffer != nullptr &&
         lookup.output.device_buffer != nullptr;
}

} // namespace
#endif

} // namespace rund::node::accel::detail
