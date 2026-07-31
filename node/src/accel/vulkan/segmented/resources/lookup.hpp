#pragma once

#include <accel/device.hpp>

#include "../../buffer/resident/batch.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct SegmentedScanBufferLookup {
  VulkanResidentBufferResult input{};
  VulkanResidentBufferResult heads{};
  VulkanResidentBufferResult output{};
};

[[nodiscard]] SegmentedScanBufferLookup
LookupSegmentedScanBuffers(const rund::AccelDevice &pick,
                           const SegmentedScanBinds &bindings) {
  SegmentedScanBufferLookup lookup{};
  VulkanResidentReq reqs[] = {
      {bindings.input, bindings.input_handle, &lookup.input},
      {bindings.heads, bindings.heads_handle, &lookup.heads},
      {bindings.output, bindings.output_handle, &lookup.output}};
  LookupVulkanResidentBatch(pick, reqs, "compute_resident_id_invalid");
  return lookup;
}

[[nodiscard]] bool
SegmentedScanLookupOk(const SegmentedScanBufferLookup &lookup) {
  return lookup.input.check.ok && lookup.heads.check.ok &&
         lookup.output.check.ok && lookup.input.device_buffer != nullptr &&
         lookup.heads.device_buffer != nullptr &&
         lookup.output.device_buffer != nullptr;
}

[[nodiscard]] const char *
SegmentedScanLookupReason(const SegmentedScanBufferLookup &lookup) {
  return !lookup.input.check.ok
             ? lookup.input.check.reason
             : (!lookup.heads.check.ok ? lookup.heads.check.reason
                                       : lookup.output.check.reason);
}

} // namespace
#endif

} // namespace rund::node::accel::detail
