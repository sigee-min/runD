#pragma once

#include <accel/device.hpp>

#include "../../buffer/resident/batch.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct GatherBufferLookup {
  VulkanResidentBufferResult values{};
  VulkanResidentBufferResult indices{};
  VulkanResidentBufferResult logical_count{};
  VulkanResidentBufferResult output{};
};

[[nodiscard]] GatherBufferLookup
LookupGatherBuffers(const rund::AccelDevice &pick,
                    const GatherBinds &bindings) {
  GatherBufferLookup lookup{};
  VulkanResidentReq reqs[] = {
      {bindings.values, bindings.values_handle, &lookup.values},
      {bindings.indices, bindings.indices_handle, &lookup.indices},
      {bindings.output, bindings.output_handle, &lookup.output},
      {bindings.logical_count, bindings.logical_count_handle,
       &lookup.logical_count}};
  LookupVulkanResidentBatch(pick, reqs,
                            bindings.logical_count_handle == nullptr ? 3u : 4u,
                            "compute_resident_id_invalid");
  return lookup;
}

[[nodiscard]] const char *GatherLookupReason(const GatherBufferLookup &lookup,
                                             const bool has_logical_count) {
  return !lookup.values.check.ok
             ? lookup.values.check.reason
             : (!lookup.indices.check.ok
                    ? lookup.indices.check.reason
                    : (!lookup.output.check.ok
                           ? lookup.output.check.reason
                           : (has_logical_count
                                  ? lookup.logical_count.check.reason
                                  : "compute_resident_id_invalid")));
}

[[nodiscard]] bool GatherLookupOk(const GatherBufferLookup &lookup,
                                  const bool has_logical_count) {
  return lookup.values.check.ok && lookup.indices.check.ok &&
         lookup.output.check.ok && lookup.values.device_buffer != nullptr &&
         lookup.indices.device_buffer != nullptr &&
         lookup.output.device_buffer != nullptr &&
         (!has_logical_count ||
          (lookup.logical_count.check.ok &&
           lookup.logical_count.device_buffer != nullptr));
}

} // namespace
#endif

} // namespace rund::node::accel::detail
