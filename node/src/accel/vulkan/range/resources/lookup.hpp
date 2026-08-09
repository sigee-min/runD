#pragma once

#include <accel/device.hpp>

#include "../../../kernel/bindings/range.hpp"
#include "../../buffer/resident/batch.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct VulkanRangeBufferLookup {
  VulkanResidentBufferResult input{};
  VulkanResidentBufferResult output{};
};

[[nodiscard]] VulkanRangeBufferLookup
LookupVulkanRangeBuffers(const rund::AccelDevice &pick,
                         const RangeBinds &bindings) {
  VulkanRangeBufferLookup lookup{};
  VulkanResidentReq reqs[] = {
      {bindings.input, bindings.input_handle, &lookup.input},
      {bindings.output, bindings.output_handle, &lookup.output}};
  LookupVulkanResidentBatch(pick, reqs, "compute_resident_id_invalid");
  return lookup;
}

[[nodiscard]] const char *
VulkanRangeLookupReason(const VulkanRangeBufferLookup &lookup) {
  return !lookup.input.check.ok ? lookup.input.check.reason
                                : lookup.output.check.reason;
}

[[nodiscard]] bool VulkanRangeLookupOk(const VulkanRangeBufferLookup &lookup) {
  return lookup.input.check.ok && lookup.output.check.ok &&
         lookup.input.device_buffer != nullptr &&
         lookup.output.device_buffer != nullptr;
}

} // namespace
#endif

} // namespace rund::node::accel::detail
