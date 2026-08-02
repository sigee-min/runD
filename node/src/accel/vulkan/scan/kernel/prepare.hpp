#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "buffers.hpp"
#include "lifetime.hpp"
#include "lookup.hpp"

#include <utility>

namespace rund::node::accel::detail {

rund::AccelCheck PrepareVulkanScan(const rund::AccelDevice &pick,
                                   const rund::kernel::ScanDesc &desc,
                                   const rund::kernel::ScanPlan &plan,
                                   const rund::kernel::ComputeDomain domain,
                                   const ScanBinds &bindings,
                                   std::shared_ptr<void> &resources,
                                   const VulkanKernelImmutablePipelines
                                       *const pipelines) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  resources.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  SetVulkanLastError(*adapter, "ok");
  if (!ScanShapeOk(desc, plan) || !ScanResidentShapeOk(plan, bindings)) {
    SetVulkanLastError(*adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  auto *const raw = new VulkanKernelScanResources{};
  std::shared_ptr<void> owned{raw, DestroyVulkanKernelScanResources};
  raw->adapter = adapter;
  raw->plan = plan;
  const rund::AccelCheck lookup = LookupVulkanScanBuffers(pick, bindings, *raw);
  if (!lookup.ok) {
    SetVulkanLastError(*adapter, lookup.reason);
    return lookup;
  }
  const rund::AccelCheck buffers =
      CreateVulkanScanScratch(*adapter, plan, *raw);
  if (!buffers.ok) {
    return buffers;
  }
  const rund::AccelCheck prepare = PrepareVulkanScanBuffers(
      *adapter, desc, plan, domain, *raw->input.device_buffer,
      *raw->output.device_buffer, *raw->logical_count.device_buffer,
      raw->totals, raw->status, raw->scan_resources,
      raw->input.ref.offset_bytes,
      raw->input.ref.count * raw->input.ref.element_bytes,
      raw->output.ref.offset_bytes,
      raw->output.ref.count * raw->output.ref.element_bytes,
      raw->logical_count.ref.offset_bytes,
      raw->logical_count.ref.count * raw->logical_count.ref.element_bytes,
      pipelines);
  if (!prepare.ok) {
    return prepare;
  }
  resources = std::move(owned);
  return rund::AccelCheck{true, "ok"};
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)bindings;
  (void)resources;
  (void)pipelines;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
