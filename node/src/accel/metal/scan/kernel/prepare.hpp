#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "buffers.hpp"
#include "lifetime.hpp"
#include "lookup.hpp"

#include <utility>

namespace rund::node::accel::detail {

rund::AccelCheck PrepareMetalScan(const rund::AccelDevice &pick,
                                  const rund::kernel::ScanDesc &desc,
                                  const rund::kernel::ScanPlan &plan,
                                  const rund::kernel::ComputeDomain domain,
                                  const ScanBinds &bindings,
                                  std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  resources.reset();
  if (!MetalPickOwnsAdapter(pick)) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (adapter == nullptr || adapter->device == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  SetMetalLastError(*adapter, "ok");
  if (!ScanShapeOk(desc, plan) || !ScanResidentShapeOk(plan, bindings)) {
    SetMetalLastError(*adapter, "compute_scan_invalid");
    return rund::AccelCheck{false, "compute_scan_invalid"};
  }
  auto *const raw = new MetalScanEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyMetalScanEncodeResources};
  raw->adapter = adapter;
  raw->desc = desc;
  raw->plan = plan;
  raw->domain = domain;
  const rund::AccelCheck lookup = LookupMetalScanBuffers(pick, bindings, *raw);
  if (!lookup.ok) {
    SetMetalLastError(*adapter, lookup.reason);
    return lookup;
  }
  const rund::AccelCheck buffers =
      AcquireMetalScanScratch(*adapter, plan, *raw);
  if (!buffers.ok) {
    return buffers;
  }
  if (!CompileMetalScanPipelines(*adapter, plan.element, raw->block,
                                 raw->prefix, raw->offset)) {
    SetMetalLastError(*adapter, "accel_metal_pipeline_unavailable");
    return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
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
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
