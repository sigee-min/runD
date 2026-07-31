#pragma once

#include <accel/device.hpp>

#include "../../buffer/resident/batch.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct ReduceBufferLookup {
  VulkanResidentBufferResult input{};
  VulkanResidentBufferResult output{};
  VulkanResidentBufferResult logical_count{};
};

[[nodiscard]] ReduceBufferLookup
LookupReduceBuffers(const rund::AccelDevice &pick,
                    const ReduceBinds &bindings) {
  ReduceBufferLookup lookup{};
  VulkanResidentReq reqs[] = {
      {bindings.input, bindings.input_handle, &lookup.input},
      {bindings.output, bindings.output_handle, &lookup.output}};
  LookupVulkanResidentBatch(pick, reqs, "compute_resident_id_invalid");
  if (bindings.logical_count_handle != nullptr) {
    VulkanResidentReq count[] = {{bindings.logical_count,
                                  bindings.logical_count_handle,
                                  &lookup.logical_count}};
    LookupVulkanResidentBatch(pick, count, "compute_resident_id_invalid");
  } else {
    lookup.logical_count = lookup.input;
  }
  return lookup;
}

[[nodiscard]] const char *ReduceLookupReason(const ReduceBufferLookup &lookup) {
  return !lookup.input.check.ok
             ? lookup.input.check.reason
             : (!lookup.output.check.ok ? lookup.output.check.reason
                                        : lookup.logical_count.check.reason);
}

[[nodiscard]] bool ReduceLookupOk(const ReduceBufferLookup &lookup) {
  return lookup.input.check.ok && lookup.output.check.ok &&
         lookup.logical_count.check.ok &&
         lookup.input.device_buffer != nullptr &&
         lookup.output.device_buffer != nullptr;
}

} // namespace
#endif

} // namespace rund::node::accel::detail
