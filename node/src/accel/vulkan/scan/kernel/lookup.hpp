#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../buffer/resident/batch.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
LookupVulkanScanBuffers(const rund::AccelDevice &pick,
                        const ScanBinds &bindings,
                        VulkanKernelScanResources &resources) {
  VulkanResidentReq reqs[] = {
      {bindings.input, bindings.input_handle, &resources.input},
      {bindings.output, bindings.output_handle, &resources.output}};
  LookupVulkanResidentBatch(pick, reqs, "compute_resident_id_invalid");
  if (bindings.logical_count_handle != nullptr) {
    VulkanResidentReq count[] = {{bindings.logical_count,
                                  bindings.logical_count_handle,
                                  &resources.logical_count}};
    LookupVulkanResidentBatch(pick, count, "compute_resident_id_invalid");
  } else {
    resources.logical_count = resources.input;
  }
  if (resources.input.check.ok && resources.output.check.ok &&
      resources.logical_count.check.ok &&
      resources.input.device_buffer != nullptr &&
      resources.output.device_buffer != nullptr &&
      resources.logical_count.device_buffer != nullptr) {
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
