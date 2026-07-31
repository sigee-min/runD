#pragma once

#include <accel/device.hpp>

#include "../../buffer/resident/batch.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct PartitionBufferLookup {
  VulkanResidentBufferResult flags{};
  VulkanResidentBufferResult values{};
  VulkanResidentBufferResult output{};
};

[[nodiscard]] PartitionBufferLookup
LookupPartitionBuffers(const rund::AccelDevice &pick,
                       const PartitionBinds &bindings) {
  PartitionBufferLookup lookup{};
  VulkanResidentReq reqs[] = {
      {bindings.flags, bindings.flags_handle, &lookup.flags},
      {bindings.values, bindings.values_handle, &lookup.values},
      {bindings.output, bindings.output_handle, &lookup.output}};
  LookupVulkanResidentBatch(pick, reqs, "compute_resident_id_invalid");
  return lookup;
}

[[nodiscard]] const char *
PartitionLookupReason(const PartitionBufferLookup &lookup) {
  return !lookup.flags.check.ok
             ? lookup.flags.check.reason
             : (!lookup.values.check.ok ? lookup.values.check.reason
                                        : lookup.output.check.reason);
}

[[nodiscard]] bool PartitionLookupOk(const PartitionBufferLookup &lookup) {
  return lookup.flags.check.ok && lookup.values.check.ok &&
         lookup.output.check.ok && lookup.flags.device_buffer != nullptr &&
         lookup.values.device_buffer != nullptr &&
         lookup.output.device_buffer != nullptr;
}

} // namespace
#endif

} // namespace rund::node::accel::detail
